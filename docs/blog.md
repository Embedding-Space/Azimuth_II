# Title to Come

## Introduction: What Were We Even Doing in There?

[Short section explainining how we've been interested in hidden space since the steering experiment (link) and we read that one paper that's in our references.bib and it was like "there's geometry to the unembedding matrix" so we decided to go exploring.]

## Initial Explorations and The Overdensity

[How about first we map the sky, like with our eyes? That could be cool, and besides eyes are really good at detecting patterns in stuff. Show sky plots. WHAMMO the overdensity appears (see 1.3a, 1.3b).]

[Norm histogram, "the spike" appears. (1.2a)]

[We use the sky maps to block out a sector of interest in r, longitude and latitude. (1.5b) Nearest-neighbor analysis (1.4f) shows huge number of tokens with very close neighbors. We build an adjacency graph, discover it's like a solid 2,212-token ball floating in a gas (1.4g).]

[Let's map it as if we were at the center. (1.6c) Let's do axial tomography (1.6d). There's something in slices 6 and 7, and in slices 7-14 it looks like there's a radially symmetric cloud of diffuse points. Let's zoom in on slices 6 and 7. (1.6f). Density is anistropic. Let's zoom in more. (1.6g). There you are, you lil sos-n-sos.]

## Is This a Problem? A Digression on Decoding

[In here we talk about how decoding works, and distinguishability through cosine similarity and logit scoring. The cluster tokens got the EXACT SAME logit scores (1.7c). So what, lots of tokens do in that middle orthogonal band. No really, these are indistinguishable even if you look directly at them (1.7e, 1.7f). This may just be a footnote, depends on how deep I want to get in to how decoding works.]

## Exploring the Overdensity

[Just what ARE these tokens? Turns out they're mostly Thai for some reason (1.7a) and in fact there's more Thai tokens in the cluster than in the rest of the vocabulary put together (1.7b). Does this mean Qwen 3 4B just sucks at Thai? No, its tokenizer doesn't output any of these tokens (1.8b)!]

[In investigating the tokenizer we discover that it only has 151,669 tokens it can output while the matrix has 151,936. These 267 literally-cannot-be-tokenized tokens are all part of the cluster.]

[Now we can form a hypothesis: These tokens are ones that never got up3dates during training. They're inert.]

[But then why would they all be close together?]

[Extended hypothesis: All the tokens in the vocabulary were initialized to this one point and these are the tokens that got left behind. Question: Are they literally all at the same point? Let's go look. NO! They are NOT all at the same point, nor are most of them at the same point. Instead there are 13 distinct points where 2 or more tokens live at the same time. (1.11a)]

[Let's look at this. (1.11c) Two points wha? bfloat16 degeneration. New hypothesis: They're not all at the same point, but they're so close together that they can only get closer by jumping to the next lattice cell.]