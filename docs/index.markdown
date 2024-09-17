---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

---
# Welcome to the Wrofacade's project!

This is the Wrofacade project's home page. You can find here the project's description,
used technology specification and more.

### Project purpose

The project goal was to develop an Aritifical Intelligence-based program which
could assist architects in their work by fast generating
a big number of tenements' facades concepts (in the form of
realistic images). Classical design applications provide an architect
with a prebuilt set of facade's elements and/or layouts. Why not
extend the number of such pieces of concepts to unlimited with generative AI?

The main challenge to be addressed was to make the application
capable of complying with requirements given by the user (an architect),
i.e. all generated facades' images must satisfy a set of conditions.
Generally, such requirements may be divided into two categories:

- **technical (hard) requirements** (e.g. facade's width and height,
number of floors)
- **style (soft) requirements** (e.g. historical style compliance, mood)

Sometimes it is desired to provide a generated image with a new facade
being similar to another (existing) facade.

Learn more on [about page]({{ site.baseurl }}{% link about.markdown %}).

### System's general concept

The concept of the pipeline of the facades generating system
is presented below (diagram created with [miro](https://miro.com/app/dashboard/)).

![pipeline-concept-diagram](img/Flowchart.jpg)

It should be noted that the concept presented above is very general - in the case
of both generative modules, there can be various implementations of them,
making use of different technologies and providing different capabilities
(e.g. support of different types of requirements passed by the user).

### Technology and implementation

Technologies that have been used:

- **split grammars** - grammar-based models can generate new facades' images
as well as new facade's segmentation masks; in such models, the generation process is more controllable
than in the case of deep generative models (like GANs) and the output shape is varying,
so this type of models provides an option to set e.g. height to width ration of the output
or the number of floors
- **deep facades segmentation models** - such models enable a user to obtain an accurate
segmentation mask of a facade the user provides
- **Generative Adversarial Networks (GANs)** - generative deep neural networks

The project is implemented in Python.

Unfortunately, not all project goals have been achieved. Having said this, authors believe
that the system architecture they designed has a great potential.

Learn more about used AI technologies [here]({{ site.baseurl }}{% link technologies.md %}).

### Contributors

**Project's authors**:

- Bianka Kowalska ([GitHub](https://github.com/bianekk))
- Hubert Baran ([GitHub](https://github.com/Hubert1225))
- Daniil Hardzetski ([GitHub](https://github.com/DanH4rd))

