# Metric Convolutions: A Unifying Theory to Adaptive Convolutions [\[Paper ICCV 2025\]](https://arxiv.org/abs/2406.05400)
![teaser_image](assets/duck_motivation_adaptive_kernels)

*** The code will be released for the ICCV 2025 conference ***

---

## Abstract

Standard convolutions are prevalent in image processing and deep learning, but their fixed kernels limits adaptability. Several deformation strategies of the reference kernel grid have been proposed. Yet, they lack a unified theoretical framework. By returning to a metric perspective for images, now seen as two-dimensional manifolds equipped with notions of local and geodesic distances, either symmetric (Riemannian) or not (Finsler), we provide a unifying principle: the kernel positions are samples of unit balls of implicit metrics. With this new perspective, we also propose *metric convolutions*, a novel approach that samples unit balls from explicit signal-dependent metrics, providing interpretable operators with geometric regularisation. This framework, compatible with gradient-based optimisation, can directly replace existing convolutions applied to either input images or deep features of neural networks. Metric convolutions typically require fewer parameters and provide better generalisation. Our approach shows competitive performance in standard denoising and classification tasks. 

---

## Installation

TODO

---

## Quick usage

TODO

---

## Demos

TODO

---


---
## License

This project is licensed under the BSDS-3 License. See the `LICENSE` file for details.

---

## Citation

If you find our work useful in your research, please cite:

```bibtex
@article{dages2024metric,
  title={Metric Convolutions: A Unifying Theory to Adaptive Convolutions},
  author={Dag{\`e}s, Thomas and Lindenbaum, Michael and Bruckstein, Alfred M},
  journal={arXiv preprint arXiv:2406.05400},
  year={2024}
}
```