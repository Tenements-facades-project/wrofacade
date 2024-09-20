---

---
# Technology specification

Here You can learn about AI technologies used to generate facades with Wrofacade.

## Split grammar learning

In this approach, we assume that structure of a facade can be modelled with
a formal grammar.

A facade's grammar represents hierarchical structure of the facade. Formally,
a (split) grammar consists of a set of terminals (representing basic parts of
a facade, e.g. a door or a window), a set of nonterminals (each facade's fragment
composed of several terminals is a nonterminal, e.g. an entire floor) and a set of productions.
A production is a function accepting one nonterminal and returning a set
of nonterminals or terminals (a production _splits_ nonterminal into terminals
or nonterminals at lower level). If a grammar describes well the hierarchical structure
of a facade, we say that the facade is the result of _a production_ of the grammar.

We can design a grammar in which some (or all) nonterminals can be split in
more than one way (i.e. exists more than one production applicable to a nonterminal).
Such a grammar is called a _nondeterministic grammar_. If for each nonterminal,
each rule applicable is assigned a probability of being chosen in a production,
then a _stochastic grammar_ is obtained. Notice that in such a grammar, nonterminals
are no longer represented with _one concrete part of a facade_, but the are rather
abstract (for instance, if one has a nonterminal representing a ground floor,
it can take various forms in the result facade, which depends on which production
rule is chosen in a production to split it and to produce further nonterminals or
terminals). Additionally, such a grammar is a _generative model_, that may be used
to generate facade's images examples.

The idea is given a set of facades' images, to build a stochastic grammar, which
terminals and nonterminals are derived from the input examples, but the grammar
is able to generate new facades examples using them.

### Grammar learning process

The input of the process is a set of facades examples, where each example
is a pair of images: (_I_, _M_); _I_ is a RGB image of a facade, and
_M_ is the segmentation mask of the facade.

The learning pipeline is as follows:

1. For each input example, build a lattice (i.e. represent the image and mask
as a grid of rectangles of different dimensions).
2. For each lattice, create a parse tree (in this project, the tool for obtaining
a parse tree from a lattice is called a _general grammar_).
3. For each parse tree, create a grammar (each leaf in the parse tree is
a terminal, and each other node in the parse tree is a nonterminal); each such a grammar
is referred to as an _instance-specific grammar_, because each one is deterministic
and produces just one input example.
4. Merge all instance-specific grammars into one grammar (which just produces
input examples with equal probabilities).
5. Iteratively perform _nonterminals merging_ - in this process, two nonterminals are chosen
and merged into _one nonterminal_, such that the grammar has lower number of nonterminals
(is "more stochastic) with possibly high grammar's likelihood on input examples.
