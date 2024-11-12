# wrofacade

This is the Wrofacade project's repository - Historic facades generation with AI, accepting user requirements.

![generated image example 1](docs/img/example-grammar-pix2pix-1.png)
![generated image example 2](docs/img/example-grammar-pix2pix-3.png)

---

**See the [project homepage](https://tenements-facades-project.github.io/wrofacade/)**

## About the project

The project goal was to develop an Aritifical Intelligence-based program which could assist architects in their work by fast generating a big number of tenements’ facades concepts (in the form of realistic images). The application should be capable of taking into account the requirements passed by the user.

The requirements on output facades can be divided into two categories:

- **technical (hard) requirements** (e.g. facade's width and height,
number of floors)
- **style (soft) requirements** (e.g. historical style compliance, mood)

## System's concept

![pipeline-concept-diagram](docs/img/concept.jpg)

Style requirements are not supported at this moment. They are planned to be introduced in the translation step.

## Supported pipelines

The primary pipeline developed in this project is (mask generator + mask to image translator).
This pipeline is represented by the class `SegmentAndTranslate`
(`src.facade_generator.pipeline.SegmentAndTranslate`). An object of this class contains
two objects: a segmentation mask generator and a mask to image translator.
Segmentation mask generator can be a grammar-based generator (class `src.segmask.grammars.GrammarMaskGenerator`)
or a segmentation model that accepts a facade image (class `src.segmask.transformers_seg.TransSegmentationHF`).

This project provides also an additional option, which is generating images using
just grammars. This pipeline is implemented by the class
`src.facade_generator.pure_grammars_generator.PureGrammarGenerator`.
