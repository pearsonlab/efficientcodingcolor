In this work at the intersection of representation learning and generative modeling, the authors use a GAN approach (BiWaveGAN) to learn a generative model of mouse USVs that comes equipped with a latent space. They then examine similarities and differences in the responses of single neurons in auditory cortex in anesthetized adult female mice. In general, analyses at the population level suggest similar responses to both classes of stimuli, though individual units tended to respond differently to the two. They argue that their method has promise for offering experimenters a large, ready supply of synthetic stimuli that match the statistics of real USVs.

I think there are many positives to note about the paper. As reflected in the citations, machine learning models have proven quite useful in studying animal vocalizations in the last several years, and an ability to replicate natural vocal stimuli is likely to be as revelatory as the widespread adoption of realistic visual stimuli has been in elucidating the function of the auditory (and related) systems. In general, the methods are well described, and the results are clearly presented. This is a solid paper that will advance the field.

However, I do have a couple of serious concerns:

1. First, GANs are notoriously brittle to train, so the applicability of the method (in particular, how difficult it was to tune hyperparameters, how important various preprocessing steps were, etc.) should be discussed. But more importantly, the justification for using GANs over, e.g., VAEs has always been that they provide higher-quality generated samples. However, the reconstructions in Figure 2 are, well, pretty bad, even in comparison to those in Figure 1 of the Goffinet paper. VAEs tend to produce blurry and oversmoothed reconstructions, but they also manage to remove some noise as a bonus, whereas the reconstruction columns in Figure 2 make serious mistakes in USV shape. For me, this undercuts much of the justification for choosing a GAN model. It could be mitigated by showing that the GAN produces a higher diversity or quality of generated USVs than a VAE, but this is not shown, making the choice to focus on reconstructions instead somewhat puzzling.

2. Along related lines, the authors mention that the generated USVs have, when slowed down, consistent and detectable auditory distortions perceptible to humans. This fact alone would make it difficult to conclude that such a difference would be imperceptible to mice (and thus to auditory cortex). Now, I don't think this is precisely the claim the authors are making, but given that the emphasis in Figure 5 is on how *similar* the two classes of stimuli are, it might produce in readers the (unlikely) impression that they are not perceptually discriminable. 

Put another way: given that the GAN itself minimizes discriminability between the real and generated stimuli, I would think it natural to ask how discriminable the two classes are *by A1 responses*. In fact, the three examples in Figure 4 suggest they are quite discriminable: the generated stimuli produce less reliable and temporally locked responses. Of course, this is only three examples -- it may not hold in the larger data set -- but I would be much more interested in an analysis that asked how easy it was to decode stimulus category from single-neuron and population neural data.

Smaller comments:
1. How were the upper bounds on correlation coefficients obtained for figure 5C? It's unclear from the paper, and it would be helpful to have a brief explanation somewhere in ll.238-241.

2. Are there any data on neural responses to latent traversals, or samples from the generator? With a model like this, data on reconstructions are helpful, but I feel the real promise of this model is the ability to systematically probe neural responses to stimuli based on generations from a latent space. It would be helpful to your story to include these data at least as a supplemental figure.

Minor comments:
- ll. 14-15: Isn't this what VAE models do? VAE models are definitely not state-of-the art for generating synthetic stimuli, but they do, in fact, learn both a latent space and have generative capability. I think this is discussed correctly in ll. 276-282 but perhaps not glossed accurately here.

- ll. 83-85: How do zero-padding and truncating compare as strategies for normalizing syllables in comparison with interpolating and rescaling?

- ll. 181-82: This detail is buried here and should probably be stated on line 3 of Algorithm 1.

- What are your thoughts on places where the neural response differs from the natural stimulus response (Fig.4 columns 1,3)? Do the responses differ proportionately to the difference between the original and reconstructed waveforms? This could be very interesting to look into.

- In Figure S1: are the BiWaveGAN spectrograms reconstructions of the USVs on the left (as the title suggests), or samples from the latent space (as suggested by the caption)? Given the current figure, this isn't entirely clear.



## Miles's review
This study used methods at the intersection of representation learning and generative modeling (BiWaveGAN) to generate realistic mouse USVs. The authors then used these generated USVs to probe properties of the right auditory cortex of female mice. They find that both reconstructions and latent traversals from BiWaveGAN create realistic-looking stimuli (in the spectral domain) that match natural USV statistics. Additionally, they show that mouse auditory cortex has similar response characteristics to natural and generated stimuli, with some subtle (but important) differences. In particular, population responses seem to be consistent across stimuli, but individual units tended to respond differently to nautral as opposed to generated stimuli.

In this study, Reilly et al. provide a promising method for generating new stimuli for neuroscience experiments. While previous generative models typically operate on spectrograms, BiWaveGAN acts directly on the audio waveform, allowing generation of new audio. While complex, the authors lay out the rationale and implementation of their model clearly, and it seems to be decently stable (as far as GANs go). The GANs were trained on spontaneous vocalizations from previous experiments, using two inbred strains of mice (C57BI/6J & DBA/2), likely all male, but unclear from the methods. Audio was presented to anaesthetized 6 female C57BL/6J mice (6-11 weeks old), while extracellular neural responses were recorded from right auditory cortex using a 32-channel Neuronexus probe. They use multiple methods to characterize responses of individual units, estimating receptive fields with Maximum Noise Entropy (MNE) models, correlations between receptive fields, comparison of learned MNE features, and sparesness measures of the population and individual units. Overall, Reilly et al. demonstrate that their method can generate realistic vocal stimuli that induce naturalistic responses in the auditory cortex -- although there are still distinct differences on the level of individual units. 

This study is, for the most part, clear and well-explained. It is a data-driven approach that efficiently utilizes advances in automated analysis methods to describe a complex set of differences in natural behavior. Reilly et al. provides a unique and novel method for investigating natural behavior, and I appreciate the effort that they have put into both making their method as easy to understand as possible and the effort it took to troubleshoot and develop this method. I believe that this study and this method will be a valuable contribution to neuroscience in general.

As follows are my suggestions, concerns, and minor comments:

Concerns & Suggestions (changes necessary for publication):
1. How were the upper bounds on correlation coefficients obtained for figure 5C? It's unclear from the paper, and it would be helpful to have a brief explanation somewhere in ll.238-241.

2. Are there any data on neural responses to latent traversals, or samples from the generator? With a model like this, data on reconstructions are helpful, but I feel the real promise of this model is the ability to systematically probe neural responses to stimuli based on generations from a latent space. It would be helpful to your story to include these data at least as a supplemental figure.

Minor comments:
* What are your thoughts on places where the neural response differs from the natural stimulus response (Fig.4 columns 1,3)? Do the responses differ proportionately to the difference between the original and reconstructed waveforms? This could be very interesting to look into.

* In Figure S1: are the BiWaveGAN spectrograms reconstructions of the USVs on the left (as the title suggests), or samples from the latent space (as suggested by the caption)? Given the current figure, this isn't entirely clear.


- ll. 14-15: Isn't this what VAE models do? VAE models are definitely not state-of-the art for generating synthetic stimuli, but they do, in fact, learn both a latent space and have generative capability.
- ll. 83-85: Zero-padding and truncating vs. interpolating and rescaling?
- ll. 181-82: This detail is buried here and should probably be stated on line 3 of Algorithm 1.
