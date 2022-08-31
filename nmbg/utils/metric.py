import numpy as np
import skimage.metrics
import torch.nn as nn
import torch
import torchvision


def _process_inputs(estimate, target, mask):
    if estimate.shape != target.shape:
        raise Exception("estimate and target have to be same shape")
    if mask is None:
        mask = np.ones(estimate.shape, dtype=np.bool)
    else:
        mask = mask != 0
    if estimate.shape != mask.shape:
        raise Exception("estimate and mask have to be same shape")
    return estimate, target, mask


def mse(estimate, target, mask=None):
    estimate, target, mask = _process_inputs(estimate, target, mask)
    m = np.sum((estimate[mask] - target[mask]) ** 2) / mask.sum()
    return m


def rmse(estimate, target, mask=None):
    return np.sqrt(mse(estimate, target, mask))


def mae(estimate, target, mask=None):
    estimate, target, mask = _process_inputs(estimate, target, mask)
    m = np.abs(estimate[mask] - target[mask]).sum() / mask.sum()
    return m


def outlier_fraction(estimate, target, mask=None, threshold=0):
    estimate, target, mask = _process_inputs(estimate, target, mask)
    diff = np.abs(estimate[mask] - target[mask])
    m = (diff > threshold).sum() / mask.sum()
    return m


class Metric(object):
    def __init__(self):
        self.reset()

    def reset(self):
        pass

    def add(self, es, ta, ma=None):
        pass

    def get(self):
        return {}

    def items(self):
        return self.get().items()

    def __str__(self):
        return ", ".join(
            ["%s=%.5f" % (key, value) for key, value in self.get().items()]
        )


class MultipleMetric(Metric):
    def __init__(self, metrics, prefix="", **kwargs):
        self.metrics = metrics
        super(MultipleMetric, self).__init__(**kwargs)
        self.prefix = prefix

    def reset(self):
        for m in self.metrics:
            m.reset()

    def add(self, es, ta, ma=None):
        for m in self.metrics:
            m.add(es, ta, ma)

    def get(self):
        ret = {}
        for m in self.metrics:
            vals = m.get()
            for k in vals:
                ret["%s%s" % (self.prefix, k)] = vals[k]
        return ret

    def __str__(self):
        lines = []
        for m in self.metrics:
            line = ", ".join(
                [
                    "%s%s=%.5f" % (self.prefix, key, value)
                    for key, value in m.get().items()
                ]
            )
            lines.append(line)
        return "\n".join(lines)


class BaseDistanceMetric(Metric):
    def __init__(self, name="", stats=None, **kwargs):
        super(BaseDistanceMetric, self).__init__(**kwargs)
        self.name = name
        if stats is None:
            self.stats = {"mean": np.mean}
        else:
            self.stats = stats

    def reset(self):
        self.dists = []

    def add(self, es, ta, ma=None):
        pass

    def get(self):
        dists = np.hstack(self.dists)
        return {
            "dist%s_%s" % (self.name, k): f(dists)
            for k, f in self.stats.items()
        }


class DistanceMetric(BaseDistanceMetric):
    def __init__(self, vec_length=1, p=2, **kwargs):
        super(DistanceMetric, self).__init__(name=str(p), **kwargs)
        self.vec_length = vec_length
        self.p = p

    def add(self, es, ta, ma=None):
        if es.shape != ta.shape or es.shape[-1] != self.vec_length:
            print("es", es.shape, "ta", ta.shape)
            raise Exception(
                "es and ta have to be of shape N x vec_length(={self.vec_length})"
            )
        es = es.reshape(-1, self.vec_length)
        ta = ta.reshape(-1, self.vec_length)
        if ma is not None:
            ma = ma.ravel()
            es = es[ma != 0]
            ta = ta[ma != 0]
        dist = np.linalg.norm(es - ta, ord=self.p, axis=1)
        self.dists.append(dist)


class PSNRMetric(BaseDistanceMetric):
    def __init__(self, max=1, **kwargs):
        super(PSNRMetric, self).__init__(name="psnr", **kwargs)
        # distance between minimum and maximum possible value
        self.max = max

    def add(self, es, ta, ma=None):
        if es.shape != ta.shape:
            raise Exception("es and ta have to be of shape Nxdim")
        if es.ndim == 3:
            es = es[..., None]
            ta = ta[..., None]
        if es.ndim != 4 or es.shape[3] not in [1, 3]:
            raise Exception(
                "es and ta have to be of shape bs x height x width x 0, 1, or 3"
            )
        if ma is not None:
            es = ma * es
            ta = ma * ta
        #import ipdb; ipdb.set_trace()
        mse = np.mean((es - ta) ** 2, axis=(1, 2, 3))
        psnr = 20 * np.log10(self.max) - 10 * np.log10(mse)
        return psnr

class SSIMMetric(BaseDistanceMetric):
    def __init__(self, data_range=None, mode="default", **kwargs):
        super(SSIMMetric, self).__init__(name="ssim", **kwargs)
        # distance between minimum and maximum possible value
        self.data_range = data_range
        self.mode = mode

    def add(self, es, ta, ma=None):
        if es.shape != ta.shape:
            raise Exception("es and ta have to be of shape Nxdim")
        if es.ndim == 3:
            es = es[..., None]
            ta = ta[..., None]
        if es.ndim != 4 or es.shape[3] not in [1, 3]:
            raise Exception(
                "es and ta have to be of shape bs x height x width x 0, 1, or 3"
            )
        if ma is not None:
            es = ma * es
            ta = ma * ta
        for bidx in range(es.shape[0]):
            ssim_list = []
            if self.mode == "default":
                ssim = skimage.metrics.structural_similarity(
                    es[bidx],
                    ta[bidx],
                    multichannel=True,
                    data_range=self.data_range,
                )
                ssim_list.append(ssim)
            elif self.mode == "deepvoxels":
                ssim = 0
                for c in range(3):
                    ssim += skimage.metrics.structural_similarity(
                        es[bidx, ..., c],
                        ta[bidx, ..., c],
                        gaussian_weights=True,
                        sigma=1.5,
                        use_sample_covariance=False,
                        data_range=1.0,
                    )
                ssim /= 3
            else:
                raise Exception("invalid mode")
            return ssim_list


