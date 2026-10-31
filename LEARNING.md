# Learning log

What I learned building this project, in my own words.

## Entry 1: Recommender basics

**What clicked**
- Matrix factorisation, from Serrano.Academy's "How does Netflix recommend movies?": every user and
  every item gets a short list of hidden taste numbers, and a user's interest in an item is how
  well the two lists line up (a dot product). The model invents the tastes itself; I never name
  them.
- Ratings can be almost useless. On Food.com 76% of starred reviews are 5 stars, so the stars
  barely separate anything. I treat "reviewed it" as the signal instead (implicit feedback).
- How the data is split decides whether a score means anything. Recommending is about the future,
  so each user's newest reviews are the test. A random split would let the model learn from the
  future.
- Popularity is a strong baseline. Most people review the same famous recipes, so "show everyone
  the bestsellers" is hard to beat, and every other model has to be judged against it.

**How it maps to this project**
- The 5-core keeps only users and recipes with at least 5 reviews. It keeps half the reviews but
  only 8% of users, because most people wrote a single review.
- Every model ranks all 41,240 recipes, not a sample of 100, so my scores look low next to papers
  that sample. NDCG@10 rewards a hit at the top more than one at position 10, and coverage shows
  whether a model only repeats the same few recipes (popularity covers 0.1% of the catalogue).
- Running the same code on MovieLens 1M first told me the evaluation works before I trusted the
  much smaller Food.com numbers.

**Still unclear**
- How companies check offline results against what users actually do (A/B tests, online metrics).
- How to pick the number of factors other than by trying a grid.

## Entry 2: Models

**What clicked**
- Item-kNN: two recipes are similar when the same people reviewed both. Cosine stops famous
  recipes from looking similar to everything, and shrinkage stops two recipes that share one
  reviewer from looking identical.
- PureSVD treats every recipe a user hasn't reviewed as a 0, as if they disliked it. BPR only asks
  that a recipe they reviewed scores above a random one they didn't, which is closer to what I
  measure (ranking).
- Writing BPR's training loop in TensorFlow myself (GradientTape, Adam, sampled negatives) made the
  loss, the gradients and the regularisation concrete instead of something a library hides.
- Overfitting shows up in the log: training loss kept going down while validation got worse.
- A recommendation built from item-kNN can explain itself, because its score is a sum over my
  picks. BPR's score is a dot product of learned numbers, so it can't point to a reason as easily.

**What tuning showed**
- Without shrinkage, item-kNN scored 0.0025 on validation because it favoured obscure recipes.
  With it, it reached about 0.014.
- BPR overfitted at a small regularisation (1e-4); 0.01 fixed it. Early stopping picked 18 epochs,
  and the test run trained for exactly 18, so test never decided anything.
- More PureSVD factors made it worse; 32 was best.
- My idea of down-weighting very heavy reviewers made results worse, so I reverted it. Their broad
  reviewing is what tells the model which recipes are generally liked.
- BPR won on validation and item-kNN won on test. I kept the settings chosen on validation;
  switching after seeing test would mean tuning on the exam.

## Entry 3: Takeaways

- The evaluation is most of the project. My best models beat popularity by only 5 to 8%, and that
  number means nothing without the bootstrap interval showing it is above zero. PureSVD's interval
  included zero, so I report it as no different.
- A small, honest result is worth more than a big number I can't defend. I set a kill rule before
  seeing results, so a negative result would still have shipped.
- The best model on a table isn't always the right product model. I use item-kNN in the demo
  because it handles a new visitor without retraining and can explain each suggestion.
- Explanations have to be true even when they look odd. Brownies were credited 58% to a cajun
  pasta. I showed the share instead of hiding it.
- Testing from a fresh clone caught a bug I would never have seen on my own machine: the tuned
  settings were git-ignored, so a clone used defaults.
- Data licences shape the product. Food.com can't be redistributed, so there is no hosted demo.
- Averages hide who a model helps. Heavy users get much better suggestions than light users with
  every model.

## Ideas for later

- Next-book recommender: richer signals (want to read, finished) and series order, a chance to try
  sequence-aware models. Would reuse this repo's evaluation code.
