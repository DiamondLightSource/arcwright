# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:27:10 2019
@author: miz23834
"""

import numpy as np
from numpy.typing import ArrayLike
import math
from typing import Optional
from pyFAI.detectors import Detector
from pyFAI.goniometer import GoniometerRefinement, ExtendedTransformation
from pyFAI.gui import jupyter
from pyFAI.units import hc  # 12.398419739640717 to convert E (keV) to Lam (Ang.^-1)
from pyFAI.containers import Integrate1dResult, Integrate2dResult
from scipy.optimize import minimize
from collections import OrderedDict
import json
from threading import Thread


class ArcModule(Detector):
    """XPDF ARC Module based on a 3x1 Medipix3 detector"""

    aliases = ["XpdfArcModule"]
    force_pixel = True
    MAX_SHAPE = (256, 768)

    def __init__(self, pixel1=55e-6, pixel2=55e-6):
        super().__init__(pixel1=pixel1, pixel2=pixel2)

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return OrderedDict((("pixel1", self._pixel1), ("pixel2", self._pixel2)))


class ArcFAI(object):
    def __init__(
        self,
        dist=0.25,
        nrj=76.69,
        n_modules=24,
        tth_offset=0.0,
        tth_offset_between_modules=3.47,
        pixel_size=55e-6,
        module_shape=(256, 768),
        pixel_pad_size=20,
        param_names_split=["dist", "poni1", "poni2", "rot1", "offset"],
        param_names_common=["scale", "nrj"],
        from_json=None,
        units="tth",
    ):
        self.dist = float(dist)
        self.nrj = nrj
        self.wavelength = hc / self.nrj * 1e-10
        self.module_shape = module_shape
        self.pixel_size = pixel_size
        self.pixel_pad_size = pixel_pad_size
        self.n_modules = n_modules
        self.module_names = ["module_{}".format(i) for i in range(self.n_modules)]
        self.tth_offset = tth_offset
        self.tth_offset_between_modules = tth_offset_between_modules
        self.module_approx_tths = self._calculate_approx_tth_values()
        self.modules = {}
        self.goniometers = {}
        self.param_names_split = param_names_split
        self.param_names_common = param_names_common
        self.max_tth_for_get_pts_per_deg = 180.0
        self.tth_values = []

        unitsdict = {
            "q": "q_A^-1",
            "q_nm": "q_nm^-1",
            "tth": "2th_deg",
            "tth_rad": "2th_rad",
            "r": "r_mm",
        }
        assert units in unitsdict, "integrate: Units '" + units + "' not supported"
        self.units = unitsdict[units]

        self.trans = ExtendedTransformation(
            dist_expr="dist",
            poni1_expr="poni1",
            poni2_expr="poni2",
            rot1_expr="rot1",
            rot2_expr="pi*(offset+scale*angle)/180.",
            rot3_expr="0.0",
            wavelength_expr="hc/nrj*1e-10",
            param_names=["dist", "poni1", "poni2", "rot1", "offset", "scale", "nrj"],
            pos_names=["angle"],
            constants={"hc": hc},
        )

        self.fit_wavelength = False

        if from_json:
            self.from_json(from_json)

        else:
            for module_name in self.module_names:
                self.modules[module_name] = ArcModule()
                self.goniometers[module_name] = self._get_module_goiniometer(
                    module_name
                )

        self.init_param()
        self.init_bounds()

        self.nt_param = lambda *x: tuple(x)
        self.nt_pos = lambda *x: tuple(x)

    def init_param(self):
        param = []
        for gonio in self.goniometers.values():
            param += list(gonio.param[: len(self.param_names_split)])
        param += list(gonio.param[len(self.param_names_split) :])
        self.param = np.array(param)

    def init_bounds(self):
        bounds = []
        for gonio in self.goniometers.values():
            bounds += list(gonio.bounds[: len(self.param_names_split)])
        bounds += list(gonio.bounds[len(self.param_names_split) :])
        self.bounds = np.array(bounds)

    def gonios_param_to_string(self):
        gonio = "module"
        p = ["dist", "poni1", "poni2", "rot1", "offset", "scale", "nrj"]
        print(
            "{: >15}\t{: >7}\t{: >7}\t{: >7}\t{: >7}\t{: >7}\t{: >7}\t{: >7}".format(
                gonio, p[0], p[1], p[2], p[3], p[4], p[5], p[6]
            )
        )
        for gonio in self.goniometers:
            p = self.goniometers[gonio].param
            print(
                "{: >15}\t{: >7}\t{: >7}\t{: >7}\t{: >7}\t{: >7}\t{: >7}\t{: >7}".format(
                    gonio, p[0], p[1], p[2], p[3], p[4], p[5], p[6]
                )
            )

    def multigonio_param_to_string(self, param=None):
        if param == None:
            param = self.param
        p = self.param_names_split
        gonio = "module"
        print(
            "{:15} {:10} {:10} {:10} {:10} {:10}".format(
                gonio, p[0], p[1], p[2], p[3], p[4]
            )
        )
        p = param
        for i, gonio in enumerate(self.module_names):
            print(
                "{:15} {:.4E} {:.4E} {:.4E} {:.4E} {:.4E}".format(
                    gonio,
                    p[5 * i + 0],
                    p[5 * i + 1],
                    p[5 * i + 2],
                    p[5 * i + 3],
                    p[5 * i + 4],
                )
            )
        for i, param_common in enumerate(self.param_names_common):
            len_param_names_split = len(self.param_names_split) * self.n_modules
            print("{}: {}".format(param_common, param[len_param_names_split + i]))

    def multigonio_param_bounds_to_string(self, param=None):
        """Need a function to easily show the parameters and bounds for each param on a gonio for all gonios on the arc"""
        pass

    def get_tth_motor_pos(self, metadata):
        return -metadata

    def _get_module_goiniometer(self, module_name, param=None):
        """Return GoniometerRefinement.

        If used without param defined, it populates the detector with approximate geometries."""
        epsilon = np.finfo(
            np.float32
        ).eps  # You can't have zero tolerance, or you throw a 'Inequality constraints incompatible' error message

        if not param:
            param = {
                "dist": self.dist,
                "poni1": 0.0,
                "poni2": (self.module_shape[1] * self.pixel_size / 2.0),
                "rot1": 0.0,
                "offset": -self.module_approx_tths[module_name],
                "scale": 1.0,
                "nrj": self.nrj,
            }

        bounds = {
            "dist": (param["dist"] - epsilon, param["dist"] + epsilon),
            "poni1": (param["poni1"] - epsilon, param["poni1"] + epsilon),
            "poni2": (param["poni2"] - epsilon, param["poni2"] + epsilon),
            "rot1": (param["rot1"] - epsilon, param["rot1"] + epsilon),
            "offset": (param["offset"] - 2.0, param["offset"] + 2.0),
            "scale": (param["scale"] - epsilon, param["scale"] + epsilon),
            "nrj": (param["nrj"] - epsilon, param["nrj"] + epsilon),
        }

        gonioref = GoniometerRefinement(
            param,
            pos_function=self.get_tth_motor_pos,
            trans_function=self.trans,
            detector=self.modules[module_name],
            wavelength=self.wavelength,
            bounds=bounds,
        )
        return gonioref

    def get_pts_per_deg(self, tth):
        """Function to determine a sensible number of points to extract per ring on each module"""
        if tth > self.max_tth_for_get_pts_per_deg:
            return 0
        else:
            return (
                max(-5.580379e-04 * tth**2 + 1.101551e-01 * tth - 1.343836e-01, 1)
            ) / 3.0

    def residu2(self, param):
        """Calculates the average of the error squared for a given parameter set

        :param param: dict, parameters dictionary for an ArcCalibrator object
        :return:      float, chi^2 for the parameter set
        """
        sumsquare = 0.0
        npt = 0
        for idx, gonio in enumerate(self.goniometers.values()):
            gonio_param = np.concatenate(
                (
                    param[
                        len(self.param_names_split)
                        * idx : len(self.param_names_split)
                        * (1 + idx)
                    ],
                    param[len(self.param_names_split) * len(self.goniometers) :],
                )
            )

            for single in gonio.single_geometries.values():
                motor_pos = single.get_position()
                single_param = self.trans(gonio_param, motor_pos)._asdict()
                if math.isnan(single_param["rot2"]):
                    single_param["rot2"] = 0.0
                pyFAI_param = [
                    single_param.get(name, 0.0)
                    for name in ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"]
                ]
                pyFAI_param.append(self.wavelength * 1e10)
                if (single.geometry_refinement is not None) and (
                    len(single.geometry_refinement.data) >= 1
                ):
                    sumsquare += single.geometry_refinement.chi2_wavelength(pyFAI_param)
                    npt += single.geometry_refinement.data.shape[0]
        return sumsquare / max(npt, 1)

    def chi2(self, param=None):
        """Calculates the average of the square of the error for a given parameter set

        :param param: dict, parameters dictionary for an ArcCalibrator object
                      if None, it gets the parameters dictionary from self.param
                      default = None
        :return:      float, chi^2 for the given parameter set
        """
        if param is not None:
            return self.residu2(param)
        else:
            return self.residu2(self.param)

    def refine2(self, method="slsqp", **options):
        """Geometry refinement tool
        See https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.minimize.html

        :param method:  name of the minimizer
        :param options: options for the minimizer
        :return:        dict, parameters dictionary for the ArcCalibrator object
        """
        if method.lower() in ["simplex", "nelder-mead"]:
            method = "Nelder-Mead"
            bounds = None
        else:
            bounds = self.bounds
        former_error = self.chi2()
        print("Cost function before refinement: %s" % former_error)
        param = np.asarray(self.param, dtype=np.float64)
        print("param array before refinement:")
        print(param)
        self.multigonio_param_to_string()
        res = minimize(
            self.residu2,
            param,
            method=method,
            bounds=bounds,
            tol=1e-12,
            options=options,
        )
        print(res)
        newparam = res.x
        new_error = res.fun
        print("Cost function after refinement: %s" % new_error)
        print(self.nt_param(*newparam))

        # print("Constrained Least square %s --> %s" % (former_error, new_error))
        if new_error < former_error:
            i = abs(param - newparam).argmax()
            if "_fields" in dir(self.nt_param):
                name = self.nt_param._fields[i]
                print(
                    "maxdelta on: %s (%i) %s --> %s"
                    % (name, i, self.param[i], newparam[i])
                )
            else:
                print("maxdelta on: %i %s --> %s" % (i, self.param[i], newparam[i]))
            self.param = newparam
            self.update_gonio_params()

            print("param array after refinement:")
            self.multigonio_param_to_string()

            # update wavelength after successful optimization: not easy
            # if self.fit_wavelength:
            #     self.wavelength = self.
        elif self.fit_wavelength:
            print("Restore wavelength and former parameters")
            former_wavelength = self.wavelength
            for sg in self.single_geometries.values():
                sg.calibrant.setWavelength_change2th(former_wavelength)
            print(self.nt_param(*self.param))
        return self.param

    def update_gonio_params(self, param=None):
        if param == None:
            param = self.param
        for i, module in enumerate(self.goniometers.values()):
            module.param = np.concatenate(
                (
                    param[
                        len(self.param_names_split)
                        * i : len(self.param_names_split)
                        * (1 + i)
                    ],
                    param[len(self.param_names_split) * len(self.goniometers) :],
                )
            )
            for sg in module.single_geometries.values():
                sg.param = np.concatenate(
                    (
                        param[
                            len(self.param_names_split)
                            * i : len(self.param_names_split)
                            * (1 + i)
                        ],
                        param[len(self.param_names_split) * len(self.goniometers) :],
                    )
                )

    def add_imgs_to_goniometer_refinements(
        self, imgs: list[ArrayLike], tths: list[float], verbose: Optional[bool] = False
    ) -> None:
        """Add one or multiple images.

        Args:
            imgs: list of image data to add.
            tths: list of two theta values
            verbose: increase output verboscity
        """
        if len(imgs) != len(tths):
            raise ValueError("must provide same number of images and tth values")
        mod_imgs = {
            tth: self.get_module_imgs_from_img(imgs[i]) for i, tth in enumerate(tths)
        }
        n = len(self.tth_values)
        for i, module_name in enumerate(self.module_names):
            for tth in tths:
                sg = self.goniometers[module_name].new_geometry(
                    "{}_tth{}".format(module_name, n),
                    image=mod_imgs[tth][i],
                    metadata=tth,
                )
        self.tth_values.extend(tths)

    def extract_cps_from_sgs(self, withPlot=False, max_tth_for_get_pts_per_deg=None):
        if max_tth_for_get_pts_per_deg:
            self.max_tth_for_get_pts_per_deg = (
                max_tth_for_get_pts_per_deg
                if max_tth_for_get_pts_per_deg == None
                else max_tth_for_get_pts_per_deg
            )
        for module in self.goniometers:
            for sg in self.goniometers[module].single_geometries.values():
                sg.extract_cp(
                    pts_per_deg=self.get_pts_per_deg(
                        sg.metadata + self.module_approx_tths[module]
                    )
                )
                if withPlot:
                    jupyter.display(sg=sg)

    def integrate1d(
        self,
        radial_range: tuple = (0, 90),
        npt: int = 9000,
        separateModules: bool = False,
        empty_value: float = np.nan,
        polarization_factor: Optional[float] = None,
        normalization_factor: Optional[float | list[float]] = None,
        mask: Optional[ArrayLike | list[ArrayLike]] = None,
        flat: Optional[ArrayLike | list[ArrayLike]] = None,
        splitpixel: bool = True,
        correctSolidAngle: bool = True,
    ) -> Integrate1dResult | tuple[Integrate1dResult, dict]:
        """Performs an integration for any data currently added to the ArcCalibrator object
        Args:
            radial_range: The radial range over which to integrate the data
            npt: The number of histogram bins into which the data are integrated
            separateModules: Whether to return the dictionary of modular results as well as the overall result
            empty_value: The value to return for a bin with no coverage
            polarization_factor: Apply polarization correction ? None: not applies. Else provide a value from -1 to +1
            normalization_factor: normalization monitors value
            mask: mask or list of masks to apply during the integration
            flat: flat or list of flats to apply during the integration
            splitpixel: whether to use splitpixel method or cython
            correctSolidAngle: whether to apply a solid angle correction
        Returns:
            res: the result of the integration
            result_dict: the individual modular results
        """
        mask = self.get_module_imgs_dict_from_img(mask)
        flat = self.get_module_imgs_dict_from_img(flat)

        threads = []
        result_dict = {}
        for module_name in self.module_names:
            process = Thread(
                target=self._integrate1dmodule,
                args=[
                    result_dict,
                    module_name,
                    radial_range,
                    npt,
                    separateModules,
                    empty_value,
                    polarization_factor,
                    normalization_factor,
                    mask[module_name],
                    flat[module_name],
                    splitpixel,
                    correctSolidAngle,
                    None,
                ],
            )
            process.start()
            threads.append(process)
        for process in threads:
            process.join()

        signal = np.zeros(npt, dtype=np.float64)
        normalization = np.zeros_like(signal)
        count = np.zeros_like(signal)
        for module_name in self.module_names:
            res_mg = result_dict[module_name]
            signal += res_mg.sum_signal
            normalization += res_mg.sum_normalization
            count += res_mg.count

        radial = res_mg.radial

        tiny = np.finfo("float32").tiny
        norm = np.maximum(normalization, tiny)

        invalid = count <= 0.0
        I = signal / norm
        I[invalid] = empty_value

        sigma = np.sqrt(I) / norm
        sigma[invalid] = empty_value

        res = Integrate1dResult(
            radial,
            I,
            sigma,
        )
        res._set_compute_engine(res_mg.compute_engine)
        res._set_unit(self.units)
        res._set_sum_signal(signal)
        res._set_sum_normalization(normalization)
        res._set_count(count)

        if separateModules:
            return res, result_dict
        else:
            return res

    def _integrate1dmodule(
        self,
        result_dict,
        module_name,
        radial_range=(0, 90),
        npt=9000,
        separateModules=False,
        empty_value=np.nan,
        polarization_factor=None,
        normalization_factor=None,
        mask=None,
        flat=None,
        splitpixel=True,
        correctSolidAngle=True,
        lst_variance=None,
    ):
        angles = [
            sg.get_position()
            for sg in self.goniometers[module_name].single_geometries.values()
        ]
        images = [
            sg.image for sg in self.goniometers[module_name].single_geometries.values()
        ]
        multigeo = self.goniometers[module_name].get_mg(
            angles
        )  # This is probably a slow step as it is building the ais
        multigeo.radial_range = radial_range
        multigeo.empty = empty_value
        multigeo.unit = self.units
        method = (
            "splitpixel" if splitpixel else "cython"
        )  # Could also be "BBox" for bbox split
        result_dict[module_name] = multigeo.integrate1d(
            images,
            npt,
            correctSolidAngle=correctSolidAngle,
            lst_variance=lst_variance,
            error_model=None,
            polarization_factor=polarization_factor,
            normalization_factor=normalization_factor,
            lst_mask=mask,
            lst_flat=flat,
            method=method,
        )  # , unit=unitsdict[units])
        return True

    def integrate2d(
        self,
        radial_range=(0, 90),
        azimuth_range=(-180, 180),
        npt=9000,
        separateModules=False,
        empty_value=np.nan,
        polarization_factor=None,
        normalization_factor=None,
        mask=None,
        flat=None,
        splitpixel=True,
    ):
        """WORK IN PROGRESS!!!
        1. Separate out to _integrate2dmodule, so that...
        2. Threading can be added
        """

        mask = self.get_module_imgs_dict_from_img(mask)
        flat = self.get_module_imgs_dict_from_img(flat)

        summed, counted, radial, azimuthal = None, None, None, None
        res_mgs = []
        for i, module_name in enumerate(self.module_names):
            angles = []
            images = []
            for sg in self.goniometers[module_name].single_geometries.values():
                angles.append(sg.get_position())
                images.append(sg.image)
            multigeo = self.goniometers[module_name].get_mg(angles)
            multigeo.radial_range = radial_range
            multigeo.azimuth_range = azimuth_range
            multigeo.empty = empty_value
            multigeo.unit = self.units
            # method = "splitpixel" if splitpixel else "cython" #Could also be "BBox" for bbox split ###NEED TO FIND 2D METHODS
            if images != []:
                res_mg = multigeo.integrate2d(
                    images,
                    npt,
                    correctSolidAngle=True,
                    lst_variance=None,
                    error_model=None,
                    polarization_factor=polarization_factor,
                    normalization_factor=normalization_factor,
                    lst_mask=mask[module_name],
                    lst_flat=flat[module_name],
                    method="splitpixel",
                )  # unit=unitsdict[units])
                res_mgs.append(res_mg)
                if summed is None:
                    summed = res_mg.sum
                    counted = res_mg.count
                else:
                    summed += res_mg.sum
                    counted += res_mg.count
                radial = res_mg.radial
                azimuthal = res_mg.azimuthal
            else:
                raise Exception(
                    "integrate2d: No single geometries / images found. Nothing to integrate!"
                )
        res = Integrate2dResult(
            summed / np.maximum(counted, 1e-10),
            radial,
            azimuthal,
            (summed**0.5) / np.maximum(counted, 1e-10),
        )
        res._set_unit(res_mg.unit)
        res._set_count(counted)
        res._set_sum(summed)
        if separateModules:
            return res, res_mgs
        else:
            return res

    def get_img_from_module_imgs(self, imgs, padvalue=np.nan):
        pad = np.full((self.pixel_pad_size, imgs[0].shape[1]), padvalue)
        img = imgs[0]
        for i in range(1, len(imgs)):
            img = np.concatenate((img, pad, imgs[i]), axis=0)
        return img

    def get_module_imgs_from_img(self, img):
        """Returns separate images for each module from a single image which has padded gaps between the modules
        :param img: A single 2D array image from the arc detector with padded gaps of self.pixel_pad_size between each module
        :return:    Array of n 2D images representing each of the n modules
        """
        return [
            img[
                (self.module_shape[0] * i + self.pixel_pad_size * i) : (
                    self.module_shape[0] * i + self.pixel_pad_size * i
                )
                + self.module_shape[0],
                :,
            ]
            for i in range(self.n_modules)
        ]

    def get_module_imgs_dict_from_img(self, img):
        if img is None:
            return {module_name: None for module_name in self.module_names}
        else:
            return {
                self.module_names[i]: img
                for i, img in enumerate(self.get_module_imgs_from_img(img))
            }

    def get_tths(self):
        return [
            sg.get_position()
            for sg in self.goniometers[self.module_names[0]].single_geometries.values()
        ]

    def getPyFaiArray(self, tths=None, method="chiArray", kwargs={}, padvalue=np.nan):
        """Produce a single image, padded for gaps between detectors, for a requested pyFAI ai method

        :param tth:      float, 2theta value (in degrees) at which to generate the array
        :param method:   string, name of method to call on a pyFAI AzimuthalIntegrator to genrate a single array
                         e.g. "chiArray", "solidAngleArray", "twoThetaArray", "rArray"
        :param padvalue: value for the padding between modules
                         default = numpy.nan
        :return:         float array, single image, padded for gaps between detectors, with the requested method values
        """
        tths = self.get_tths() if tths == None else tths
        imgs = [
            [
                getattr(self.goniometers[module_name].get_mg([tth]).ais[0], method)(
                    **kwargs
                )
                for module_name in self.module_names
            ]
            for tth in tths
        ]
        return [self.get_img_from_module_imgs(img, padvalue=padvalue) for img in imgs]

    def solidAngleArray(self, tths=None, kwargs={"absolute": True}, padvalue=np.nan):
        return self.getPyFaiArray(
            tths, method="solidAngleArray", kwargs=kwargs, padvalue=padvalue
        )

    def chiArray(self, tths=None, padvalue=np.nan):
        return self.getPyFaiArray(tths, method="chiArray", padvalue=padvalue)

    def twoThetaArray(self, tths=None, padvalue=np.nan):
        return self.getPyFaiArray(tths, method="twoThetaArray", padvalue=padvalue)

    def rArray(self, tths=None, padvalue=np.nan):
        return self.getPyFaiArray(tths, method="rArray", padvalue=padvalue)

    def calc_pos_zyx(self, tths=None, corners=True, padvalue=np.nan):
        """Calculates arrays for z, y and x positions of each pixel, padded for gaps between detectors

        :param tth:      float, 2theta value (in degrees) at which to generate the array
        :param padvalue: value for the padding between modules
                         default = numpy.nan
        : return:        array [z,y,x] images, padded for gaps between detectors, with the z, x, y pixel locations

        WORK IN PROGRESS.
        1. calc_pos_zyx returs a 4th dimension that we are currently ignoring!
        2. currently part way through implementing calculation for all tths on the object
        """
        tths = self.get_tths() if tths == None else tths
        #        imgs = np.asarray([self.goniometers[module_name].get_mg([tth]).ais[0].calc_pos_zyx(corners=corners) for module_name in self.module_names])
        #        z = [i[0] for i in imgs[:,:,:,:,0]]
        #        y = [i[1] for i in imgs[:,:,:,:,0]]
        #        x = [i[2] for i in imgs[:,:,:,:,0]]
        return None  # [self.get_img_from_module_imgs(z),self.get_img_from_module_imgs(y),self.get_img_from_module_imgs(x)]

    def cos_incidence(self, tth, path="cython", padvalue=np.nan):
        x = np.arange(self.module_shape[0])
        y = np.arange(self.module_shape[1])
        YY, XX = np.meshgrid(y, x)
        imgs = [
            self.goniometers[module_name]
            .get_mg([tth])
            .ais[0]
            .cos_incidence(XX, YY, path=path)
            for module_name in self.module_names
        ]
        return self.get_img_from_module_imgs(imgs, padvalue=padvalue)

    def to_json(self, filepath=None):
        """Creates a json object representating the ArcCalibrator object parameters

        :param filepath: string, filepath to save the json file to
                         if None, it will just return the json dictionary and not write the file
                         default = None
        :return:         dict, json dictionary with the ArcCalibrator object parameters
        """
        param_common = {
            param_name: self.param[
                len(self.param_names_split) * len(self.goniometers) + i
            ]
            for i, param_name in enumerate(self.param_names_common)
        }
        goniometers = {}
        for module in self.goniometers:
            param = {
                param_name: self.goniometers[module].param[i]
                for i, param_name in enumerate(self.param_names_split)
            }
            goniometers[module] = param
        json_dict = {
            "wavelength": self.wavelength,
            "param_common": param_common,
            "goniometers": goniometers,
        }
        if filepath:
            with open(filepath, "w") as outfile:
                json.dump(json_dict, outfile, indent=4)
        return json_dict

    def _clear(self) -> None:
        """Clear the stored modules, goniometers, tth values."""
        self.modules = {}
        self.goniometers = {}
        self.module_approx_tths = {}
        self.n_modules = 0
        self.module_names = []

    def _calculate_approx_tth_values(self) -> dict:
        """Calculates the approx tth values using the tth_offset_between_modules
        Returns:
            dictionary mapping module names to approximate tth values
        """
        return {
            "module_{}".format(i): self.tth_offset
            + (i * self.tth_offset_between_modules)
            for i in range(self.n_modules)
        }

    def from_json(self, injson: str | dict, nrj: Optional[float] = None) -> None:
        """Populates an ArcCalibrator object with parameters from a json file.

        Args:
            injson: path to a valid json file, or a valid dictionary
            nrj: The energy in keV if you want to overwrite the value from the configuration
        """
        if type(injson) == type({}):
            self.json = injson
        else:
            with open(injson) as json_file:
                self.json = json.load(json_file)
        if nrj == None:
            self.wavelength = self.json["wavelength"]
            self.nrj = self.json["param_common"]["nrj"]
        else:
            assert nrj in [
                40.05,
                65.40,
                76.69,
            ], "from_json: nrj {} keV not recognised as a valid energy".format(nrj)
            if self.json["param_common"]["nrj"] != nrj:
                print(
                    "WARNING: You are overwriting the calibration file energy of {} keV with a new value of {} keV)".format(
                        self.json["param_common"]["nrj"], nrj
                    )
                )
            self.nrj = nrj
            self.wavelength = (hc / nrj) * 1e-10
        self._clear()
        self.n_modules = len(self.json["goniometers"])
        for module_name, param in self.json["goniometers"].items():
            self.module_names.append(module_name)
            self.modules[module_name] = ArcModule()
            param.update(self.json["param_common"])
            self.goniometers[module_name] = self._get_module_goiniometer(
                module_name, param
            )
            self.module_approx_tths[module_name] = -param["offset"]
