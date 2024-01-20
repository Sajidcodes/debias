**Diagnosing and Debiasing Large Language Models and Respective Embeddings**

**Table of Contents**
1. Abstract
2. Introduction
3. Objectives
4. Techniques of measuring bias
5. Debiasing Techniques
6. Results
7. Limitations
8. References

   
**Abstract**
Pre-trained language models like BERT have achieved state-of-the-art performance on many
natural language processing tasks. However, these models often perpetuate and amplify
societal biases present in their training data related to gender, race, and culture. This research
aims to conduct bias identification and implement bias mitigation techniques for language
models. We comprehensively analyze major models to uncover different forms of bias, using
testing frameworks that expose biases in word associations, sentence completions, sentiment
analysis and entity/event coverage. We then develop novel model-agnostic debiasing algorithms
focused on balancing word embeddings and augmenting transformer self-attention mechanisms
during training. Experiments measure the impact of our debiasing approach on task
performance. This research will expand literature on quantifying and addressing biases in
natural language AI systems. We discuss limitations and future work needed to develop truly fair
and inclusive language technologies.

**Introduction**
Large pre-trained language models have proven effective across a variety of tasks in natural
language processing, often obtaining state of art performance. These models are trained on a
large number of texts that originated from unmoderated sources, such as the internet. While the
performance of these pre trained models is remarkable, they tend to capture social biases from
the data which they are trained on. An increasing amount of research has focused on
developing techniques to mitigate these biases.
This project focuses on diagnosing and debiasing techniques, to contribute to the development
of fairer and more inclusive natural language processing (NLP) models.

**Objectives**
Identification of Biases: Conduct a comprehensive analysis to identify various types of biases
present in existing language models. This includes but is not limited to gender, race, and cultural
biases.
Measure the effects of debiasing on the overall quality and performance of sentence
embeddings.
Implementation of De-biasing Strategies: Implement and test a combination of techniques to
minimize biases within the language models.

**Techniques of Measuring Bias**
There are three intrinsic bias techniques to evaluate debiasing techniques. We select these
benchmarks as they are used to measure not only gender bias but also racial and religious bias
in language models.

1. **Sentence Encoder Association Test (SEAT)**: SEAT is an extension of Word Embedding
Association Test (WEAT).
Four-word sets are used by WEAT: two sets of target words and two sets of bias attribute words.
A kind of bias is characterized by the attribute word sets. One way to test for gender bias is to
utilize the attribute word sets {man, he, him,...} and {woman, she, her,...}. Specific notions are
characterized by the target word sets. To define the ideas of family and career, respectively, the
target word sets {family, child, parent,...} and {work, office, profession,...} could be utilized.
WEAT assesses whether there is a tendency for word representations from a given target word
set to be more closely connected with word representations from a given attribute word set. If,
for example, the depictions of the female attribute terms mentioned above tended to be more
closely associated with the representations for the family target words, this may be indicative of
bias within the word representations.
Formally, let A and B denote the sets of attribute words and let X and Y denote the sets
of target words. The SEAT test statistic is
s(X, Y, A, B) = X x∈X s(x, A, B) − X y∈Y s(y, A, B)
where for a particular word w, s(w, A, B) is de[1]fined as the difference between w’s mean
cosine similarity with the words from A and w’s mean cosine similarity with the words from B
s(w, A, B)= 1 |A| X a∈A cos(w, a)− 1 |B| X b∈B cos(w, b).
They report an effect size given by
d = μ({s(x, A, B)}x∈X) − μ({s(y, A, B)}y∈Y ) σ({s(t, X, Y )}t∈A∪B)
where μ denotes the mean and σ denotes the standard deviation. Here, an effect size closer to
zero is indicative of a smaller degree of bias in the representations. To create a sentence-level
version of WEAT (re[1]ferred to as SEAT substitute the attribute words and target words from
WEAT into synthetic sentence templates (e.g., “this is a [WORD]”) to create a collection of
sentences. Now, given sets of sentences containing attribute and target words, the WEAT test
statistic can be computed using sentence-level representations obtained from a pre-trained
language model.

2. **Stereo Set:** four different forms of stereotype bias in language models are measured using a
crowdsourced dataset. A context sentence, such as "our housekeeper is [MASK]," is included in
each Stereo Set example, along with three potential associations (completions) for the
sentence: one is stereotypical, one is anti-stereotypical, and the third is unrelated. In the above
example, "our housekeeper is Mexican" might be a stereotype association, "our housekeeper is
American" an anti-stereotypical association, and "our housekeeper is computer" an
unconnected association. We score the stereotypical and anti-stereotypical associations for
every example that is included in a language model in order to determine how biased the model
is. Next, we determine what proportion of cases a model Favors over the anti-stereotypical
association, or the stereotypical association. This percentage is what we refer to as a model's
stereotype score. Another indicator of a model's aptitude for language modeling is Stereo Set.
Furthermore, we rate the irrelevant association for every sample in the dataset. Next, we
calculate the proportion of examples for which a model favors the relevant association over the
unrelated association (either the stereotypical association or the anti-stereotypical link). This
percentage is what we refer to as a model's language modeling score. We compare our
debiased models with the test set from the Stereo Set. We assess debiased models for every
domain by comparing them to the corresponding Stereo Set test set split (for example, gender
debiased models are assessed against the examples of gender prejudice).

3. **Crowdsourced Stereotype Pairs(CrowS-pairs):** A crowdsourced dataset called
CrowS-Pairs is made up of pairs of minimally remote phrases, or sentences that differ only in a
tiny number of characters. Each pair's opening statement presents a caricature of a historically
marginalized American group. For instance, the statement "alcoholics reside in trailers" may
represent a socioeconomic stereotype. The stereotype that was presented in the first sentence
of each pair is then broken in the second sentence. For instance, the statement "People who
live in mansions are alcoholics" contradicts the first line, or rather, it is the opposite of the
stereotype.
By counting the number of times a model chooses the stereotyped sentence over the
anti-stereotypical sentence in each pair, we may determine how biased a language model is.
In a manner akin to StereoSet, we assign a score to every pair of sentences based on
concealed token probabilities. We calculate the masked token probability of the tokens specific
to the stereotypical sentence for each pair of sentences in order to score the stereotypical
sentence. We would calculate the masked token probability of trailers in the aforementioned
example. Every anti-stereotypical sentence receives the same score. In the event that a
sentence contains more than one unique token, we calculate the average masked token
probability by masking each distinct token separately. The percentage of examples for which a
model assigns a higher masked token probability to the stereotypical sentence as opposed to
the anti-stereotypical sentence is known as the stereotype score of the model.
**Debiasing Techniques**
1. Contrafactual Data Augmentation : is a data-driven debiasing technique that's
frequently applied to reduce gender bias. The general idea behind CDA is to rebalance a
corpus by changing bias attribute words (like he/she) in a dataset. For instance,
changing the phrase "the doctor went to the room and he grabbed the syringe" to "the
doctor went to the room and she grabbed the syringe" could help reduce gender bias.
After that, the rebalanced corpus is frequently utilized for additional model debiasing
training. Although gender debiasing has been the primary application of CDA, we also
assess its efficacy for other kinds of biases. To generate counterfactual examples, we
can swap out religious terms in a corpus, such as "church" with "mosque," in order to
create CDA data for mitigating religious bias.
2. **Dropout:** The use of dropout regularization (Srivastava et al., 2014) as a bias mitigation
strategy is examined by Webster et al. (2020). They look into doing an extra pre-training
phase and raising the dropout parameters for the attention weights and hidden
activations of BERT and ALBERT. Through experimentation, they discover that in these
models, a higher dropout regularization decreases gender bias. They postulate that
dropouts are able to prevent themselves from learning unfavorable word associations by
disrupting the attention mechanisms in BERT and ALBERT. We expand this research to
include additional bias kinds. We use increased dropout regularization to conduct an
additional pre-training phase on sentences from English Wikipedia, in a manner similar
to CDA.
3. **Self Debias:** A post-hoc debiasing method is put forth by S. Schick et al. (2021) that
makes use of a model's internal knowledge to dissuade it from producing biased text.
Informally, Schick et al. (2021) suggest that a model be first encouraged to generate
toxic text by hand-crafted prompts. An autoregressive model might, for instance, prompt
generation to respond, "The following text discriminates against people because of their
gender." The model can then be used to generate a second, non-discriminative
continuation in which the likelihood of tokens considered likely during the first toxic
generation is reduced.
Crucially, Self-Debias does not change a model's internal representations or parameters
because it is a post-hoc text generation debiasing procedure. Therefore, for downstream
NLU tasks, Self-Debias cannot be used as a bias mitigation strategy.
Informally, Schick et al. (2021) suggest starting with a model that is encouraged to
produce toxic text by hand-crafted prompts. An autoregressive model might, for instance,
prompt generation to read, "The following text discriminates against people because of
their gender." The model can then be used to generate a second, non-discriminative
continuation in which the likelihood of tokens considered likely during the first toxic
generation is reduced.
Crucially, Self-Debias does not change a model's internal representations or parameters
because it is a post-hoc text generation debiasing procedure. Therefore, Self-Debias
cannot be applied to downstream NLU tasks (e.g., GLUE) as a bias mitigation strategy.
Furthermore, we are unable to assess SEAT because it measures bias in a model's
representations, whereas Self-Debias does not change a model's internal
representations.
4. **Sentence Debias :** In 2020, Liang et al. expand Bolukbasi et al. (2016) introduced the
word embedding debiasing technique known as Hard-Debias representations of
sentences. Sentence Debias is a projection-based method of debiasing that needs
estimating a linear subspace in relation to a specific kind of prejudice. Examples of
sentence representations projected onto the estimated bias to become debiased
subspace and deducting the projection that results from the sentence's original
representationLiang et al. (2020) compute a bias subspace using a three-step process.
They start by defining a list of words with bias attributes (like he/she). Secondly, they
incorporate words with biased attributes into sentences. This is accomplished by locating
instances of the bias attribute words in sentences across a corpus of text. Using CDA,
two sentences that differ only in the bias attribute word are produced for every sentence
that was discovered during this contextualization step. Lastly, the bias subspace is
estimated. A pre-trained model can yield a corresponding representation for every
sentence that was acquired during the contextualization step. Following that, the
principal directions of variation of the resulting data are estimated using Principal
Component Analysis (PCA; Abdi and Williams 2010).
5. **Iterative Null Space Projection(INLP):** Similar to SentenceDebias, INLP is a
projection-based debiasing technique proposed by Ravfogel et al. (2020).By teaching a
linear classifier to predict the protected property you wish to remove (like gender) from the
representations, INLP, in general, debiases the representations of a model. Subsequently,
representations can be debiased by projecting them into the weight matrix's null space of
the learned classifier. This effectively eliminates from the representation all of the data that
the classifier used to predict the protected attribute. The representation can then be
debiased by repeating this procedure.In our experiments, we identify instances of bias
attribute words (e.g., he/she) in English Wikipedia in order to create a classification
dataset for INLP.For instance, we categorize every sentence from the English Wikipedia
according to one of three classes for gender bias based on whether a sentence contains a
male word, a female word, or no gendered words.
Technique which is more effective in mitigating bias
We found Self-Debias to be the strongest debiasing technique. Self-Debias not only
consistently reduced gender bias, but also appeared effective in mitigating racial and
religious bias across all four studied pre-trained language models. Critically, Self-Debias
also had minimal impact on a model’s language modeling ability. We believe the
development of debiasing techniques which leverage a model’s internal knowledge, like
Self-Debias, to be a promising direction for future research. Importantly, we want to be
able to use “self-debiasing” methods when a model is being used for downstream tasks
**Limitations**
1) We only investigate bias mitigation techniques for language models trained in English.
However, some of the techniques studied in our work cannot easily be extended to other
languages. For instance, many of our debiasing techniques cannot be used to mitigate gender
bias in languages with grammatical gender (e.g., French).
2)Our work is skewed towards North American social biases.
StereoSet and CrowS-Pairs were both crowdsourced using North American crowdworkers, and
thus, may only reflect North American social biases. We believe analyzing the effectiveness of
debiasing techniques cross-culturally to be an important area for future research. Furthermore,
all of the bias benchmarks used in this work have only positive predictive power. For example, a
perfect stereotype score of 50% on StereoSet does not indicate that a model is unbiased.
3) Many of our debiasing techniques make simplifying assumptions about bias.
For example, for gender bias, most of our debiasing techniques assume a binary definition of
gender. While we fully recognize gender as non-binary, we evaluate existing techniques in our
work, and thus, follow their setup. Manzini et al. (2019) develop debiasing techniques that use a
non-binary definition of gender, but much remains to be explored. Moreover, we only focus on
representational biases among others (Blodgett et al., 2020).
Conclusion
To the best of our knowledge, we have carried out the first extensive assessment of various
debiasing methods for language models that have already been trained. We examined the
effectiveness of every debiasing method in reducing racial, gender, and religious bias in four
language models that have already been trained: BERT, ALBERT, RoBERTa, and GPT-2. In
addition to assessing how well each debiasing method mitigates bias, we also looked into the
effects of debiasing on language modeling and downstream NLU task performance using three
intrinsic bias benchmarks. Our goal is to improve the direction of upcoming bias mitigation
research with our work.
**References**
1. Yue Guo, Yi Yang, Ahmed Abbasi (2022). Auto-Debias: Debiasing Masked Language
Models with Automated Biased Prompts. In Proceedings of the 60th Annual Meeting of the
Association for Computational Linguistics Volume1: Long Papers, pages 1012-1023 May 22-27,
2022 Association for Computational Linguistics
2. Nicholas Meade, Elinor Poole-Dayan, Siva Reddy (2022). An Empirical Survey of the
Effectiveness of Debiasing Techniques for Pre-trained Language Models
3. Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton, Yoav Goldberg (2020). Null It
Out: Guarding Protected Attributes by Iterative Null Space Projection. In Proceedings Of The
58th Annual Meeting of the Association for Computational Linguistics, pages 7237–7256 July
5-10,2020. Association for Computational Linguistics.
