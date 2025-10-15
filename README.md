# Metric Convolutions: A Unifying Theory to Adaptive Convolutions [\[Paper ICCV 2025\]](https://openaccess.thecvf.com/content/ICCV2025/html/Dages_Metric_Convolutions_A_Unifying_Theory_to_Adaptive_Image_Convolutions_ICCV_2025_paper.html)
![teaser_image](assets/duck_motivation_adaptive_kernels.png)


## Poster

![poster](assets/Metric_Convolutions_Poster_ICCV_2025.png)

---

## Abstract

Standard convolutions are prevalent in image processing and deep learning, but their fixed kernels limits adaptability. Several deformation strategies of the reference kernel grid have been proposed. Yet, they lack a unified theoretical framework. By returning to a metric perspective for images, now seen as two-dimensional manifolds equipped with notions of local and geodesic distances, either symmetric (Riemannian) or not (Finsler), we provide a unifying principle: the kernel positions are samples of unit balls of implicit metrics. With this new perspective, we also propose *metric convolutions*, a novel approach that samples unit balls from explicit signal-dependent metrics, providing interpretable operators with geometric regularisation. This framework, compatible with gradient-based optimisation, can directly replace existing convolutions applied to either input images or deep features of neural networks. Metric convolutions typically require fewer parameters and provide better generalisation. Our approach shows competitive performance in standard denoising and classification tasks. 

---

## Installation

There is no need to for fancy import packages. The usual basic ones should do. Our code is in torch.

---

## Demos

You can find demos on how to use metric convolutions in the src folder.

---

## License

This project is licensed under the BSDS-3 License. See the `LICENSE` file for details.

---

## Citation

If you find our work useful in your research, please cite:

```bibtex
@InProceedings{Dages_2025_ICCV,
    author    = {Dag\`es, Thomas and Lindenbaum, Michael and Bruckstein, Alfred M.},
    title     = {Metric Convolutions: A Unifying Theory to Adaptive Image Convolutions},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {13974-13984}
}
```