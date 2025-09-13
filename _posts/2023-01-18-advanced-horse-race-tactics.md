---
layout: post
title: Advanced Horse Race Tactics using Trakus Coordinate Data
description: The impact of drafting percentage, path efficiency, race strategy and speed fluctuation on horse race tactics.
authors: [tlyleung, plyleung]
canonical_url: "https://www.kaggle.com/tlyleung/advanced-horse-race-tactics-using-coordinate-data"
x: 64
y: 41
---

> **Authors' Note**
>
> This post first appeared as a [Kaggle notebook submission](https://www.kaggle.com/code/tlyleung/advanced-horse-race-tactics-using-coordinate-data/notebook) for the [Big Data Derby 2022](https://www.kaggle.com/competitions/big-data-derby-2022) analytics competition on 10 November 2022, where it won the 3<sup>rd</sup> place prize.

## Executive Summary

In this report, we will:

1. Extract four performance-impacting features (drafting proportion, path efficiency, race strategy, speed fluctuation) from the coordinate data
2. Analyse these features in relation to race distance and race surface/going
3. Determine the predictive power of these features using a Multinomial Logit model
4. Recommend racing strategies which we think will help horse owners, trainers and jockeys

> **Key Takeaways**
>
> - Jockeys and trainers should choose race tactics that prioritise **increasing path efficiency** and **decreasing speed fluctuation** over drafting percentage and racing strategy.
> - **Public odds significantly undervalue horses with lower speed fluctuations.** If these undervalued horses can be identified before the race, this inefficiency can be exploited by placing exotic bets covering these horses.

## Introduction

The most groundbreaking information provided in the NYTHA/NYRA dataset is the Trakus coordinate data provided for each horse during the race. Our focus is to extract features from the coordinate data which will help horse owners, trainers and jockeys make better decisions on their racing strategies.

The four features we will generate and analyse are the drafting proportion, path efficiency, race strategy (early runner or sustainer, etc.) and speed fluctuation of a horse during a race. Although horse trainers and jockeys may not be able to control exactly how much drafting benefit they receive, jockeys can certainly attempt to choose strategies that offer more drafting opportunities.

We supplement the Trakus coordinate dataset with the [Big Data Derby 2022: Global Horse IDs and places dataset](https://www.kaggle.com/datasets/themarkgreen/big-data-derby-2022-global-horse-ids-and-places) compiled by Mark Green.

## Data Preprocessing

### Coordinate Reference System Transformation

We use GeoPandas to transform the Trakus coordinate data from the latitude/longitude coordinate reference system based on the Earth’s center of mass that uses degrees as its measurement unit (ESPSG:4326) to a coordinate reference system centered on New York Long Island that uses metres as its measurement unit (EPSG: 32118). This transformation enables more accurate distance calculations.

### Finish Line Truncation

The first Trakus sample provided for each race represents the starting position of the horses at the beginning of the race run-up. However, the final Trakus sample of each race does not represent the point where the horses cross the finish line, but is quite some time after the race finishes. The coordinates of the horses after they cross the finish line are irrelevant because the race has already finished.

To remove all coordinates where the horse has already finished the race, we sketch each horse’s path, intersect it with the finish line, and remove the coordinates past the finish line. To determine the location of the finish line, we pinpointed the exact coordinate location of the line on Google Maps.

This is what it looks like:

{% include figures/posts/advanced-horse-race-tactics/finish-line-trucation-without.html %}

{% include figures/posts/advanced-horse-race-tactics/finish-line-trucation-with.html %}

## Feature Engineering

We outline four features generated from the Trakus coordinate data that we use to determine horse performance:

- **Drafting Percentage:** percentage of the race where a horse has a drafting benefit.
- **Racing Strategy:** whether a horse prefers to lead from the front, settle in the middle of the pack, or start slow.
- **Path Efficiency:** the efficiency of the path the horse takes around the course.
- **Speed Fluctuation:** the amount of variation in velocity the horse experiences once the horse has settled into the race.

### How much drafting benefit does each horse receive?

To generate a feature representing drafting benefit, we need to look at the fundamentals of drafting. The reduction of aerodynamic drag from drafting is an essential element of horseracing because horses are able to save energy which they can expend closer to the finish line. Drafting occurs when a horse follows closely behind another horse.

The coordinate data helps us look slightly ahead of each horse to see if another horse occupies that space. We draw a circle 3 metres ahead of the horse’s current coordinate position and if another horse’s coordinate lies inside the 2 metre radius circle, then there is a drafting benefit.

In the figure below:

- red points represent horses _with_ draft benefit
- black points represent horses _without_ draft benefit
- yellow circles represent look-ahead regions used to check for the presence of drafting

{% include figures/posts/advanced-horse-race-tactics/features-drafting-benefit.html %}

Using this method, we can determine whether a horse receives a drafting benefit for every coordinate of a race. Hence, we can calculate the percentage of the race where a horse is receiving a drafting benefit. This feature takes a value between 0 and 1 inclusive, where 0 means that a horse received no drafting benefit during the race and 1 means that a horse received a drafting benefit for 100% of the race.

### What racing strategy is used by each horse?

To determine the racing strategy of each horse, we look at each horse’s running position once the race settles. This is usually determined at the first call of the race which is approximately 20 seconds into the race. The assumption is that the horses are still fresh near the start of the race so they can implement their strategy and position themselves ideally at the first call. At the first call, front runners would have a lower running position while horses that start slow and finish quickly would have a higher running position.

However, we are not given the running position of each horse so we will estimate the running position of the horses by calculating the angle of each coordinate from the centroid of the racetrack.

{% include figures/posts/advanced-horse-race-tactics/features-racing-strategy.html %}

We also normalise the value of the running position so that the feature is easier to compare between races with a different number of participants. After the normalisation, our racing strategy feature takes a value between 0 and 1 inclusive, where 0 means that the horse was leading 20 seconds into the race and 1 means that the horse was in last place 20 seconds into the race.

### How efficient is the horse’s path?

A horse’s path is more efficient when it travels close to the rail because it travels less distance. However, this might not always be the best choice because the path closest to the rail might be congested with other horses, so it might make more sense to choose a less efficient path by going around the other horses. Also, the path close to the rail might be more muddy when it rains, which means that the shortest path might no longer be the ideal path.

To produce this feature of path efficiency, we first calculate the distance the horse travels by converting its path coordinates into segment distances (with the help of an improved Vincenty formula that uses geodesic distances) and adding them up. Then, in order to create a path efficiency metric we take the official race distance and divide it by the distance travelled by the horse, so a path efficiency of 1 means that the horse travelled the official distance of the race and 0.5 means that the horse travelled double the official distance of the race.

Here is a visual representation of a race consisting of a starter with a higher path efficiency (in blue) compared to one in the same race with a lower path efficiency (in orange):

{% include figures/posts/advanced-horse-race-tactics/features-path-efficiency.html %}

### How consistent is the horse’s speed?

Horses with a smooth race usually have a low speed fluctuation because they have not needed to slow down suddenly due to being blocked by other horses. These unnecessary speed changes, in particular deceleration, cause horses to exert an unnecessary amount of energy which could be used to maintain or even increase their speed instead.

To generate this feature, we calculate the horse’s speed for each race segment by getting the distance travelled divided by the Trakus time interval, which is 0.25 seconds. Afterwards, we compute the acceleration by taking the difference between the speeds of consecutive segments. Finally, we take the standard deviation across the horse’s acceleration to determine the fluctuation of the horse’s speed. We only start measuring a horse’s speed fluctuation after the first call because the start of the race can be quite hectic as horses accelerate to their maximum speed while jostling for their ideal position.

{% include figures/posts/advanced-horse-race-tactics/features-speed-fluctuation-velocity.html %}

{% include figures/posts/advanced-horse-race-tactics/features-speed-fluctuation-acceleration.html %}

For this feature, a value of 0 indicates that the horse has no speed inconsistencies and has kept a perfectly consistent speed from the first call to the finish line, while a value of 0.5 means that the horse’s segment accelerations have a standard deviation of 0.5.

## Feature Analysis

### Attributes of Winning Horses

By analysing the race winners, we better understand about the attributes of race winners which might help us make informed decisions on how to become a race winner. We have segmented this analysis by race distance and race surface/going because different facets may favour specific race strategies (e.g. longer distances may favour slow starters over front runners).

#### Drafting Percentage

<figure>
  <figcaption>Win % by drafting percentage vs. distance</figcaption>
  <div class="mb-3 flex flex-row flex-wrap gap-x-4 text-sm">
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#cb3300" width="15" height="15"></rect></svg><span>0–20%</span></div>
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#3163ce" width="15" height="15"></rect></svg><span>20–40%</span></div>
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#ff9b00" width="15" height="15"></rect></svg><span>40–60%</span></div>
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#02991d" width="15" height="15"></rect></svg><span>60–80%</span></div>
  </div>
  <img src="/assets/images/posts/advanced-horse-race-tactics/drafting_percentage_distance.svg" alt="Win % by drafting percentage vs. distance">
</figure>

<figure>
  <figcaption>Win % by drafting percentage vs. surface/going</figcaption>
  <div class="mb-3 flex flex-row flex-wrap gap-x-4 text-sm">
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#cb3300" width="15" height="15"></rect></svg><span>0–20%</span></div>
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#3163ce" width="15" height="15"></rect></svg><span>20–40%</span></div>
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#ff9b00" width="15" height="15"></rect></svg><span>40–60%</span></div>
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#02991d" width="15" height="15"></rect></svg><span>60–80%</span></div>
  </div>
  <img src="/assets/images/posts/advanced-horse-race-tactics/drafting_percentage_surface_going.svg" alt="Win % by drafting percentage vs. surface/going">
</figure>

> **Observation**
>
> Horses with a meager drafting percentage have a much higher chance of winning the race than other horses. This is likely because these particular horses are front-runners with no one ahead. Interestingly, the win percentage of horses with a high drafting percentage increases for longer distances which means that a high drafting strategy is more effective for longer distances. Also, the low drafting percentage advantage is especially significant when the going is fast or firm but less significant when the going is more wet.

#### Path Efficiency

<figure>
  <figcaption>Win % by path efficiency vs. distance</figcaption>
  <div class="mb-3 flex flex-row flex-wrap gap-x-4 text-sm">
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#cb3300" width="15" height="15"></rect></svg><span>98.0–98.5%</span></div>
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#3163ce" width="15" height="15"></rect></svg><span>98.5–99.0%</span></div>
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#ff9b00" width="15" height="15"></rect></svg><span>99.0–99.5%</span></div>
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#02991d" width="15" height="15"></rect></svg><span>99.5–100.0%</span></div>
  </div>
  <img src="/assets/images/posts/advanced-horse-race-tactics/path_efficiency_distance.svg" alt="Win % by path efficiency vs. distance">
</figure>

<figure>
  <figcaption>Win % by path efficiency vs. surface/going</figcaption>
  <div class="mb-3 flex flex-row flex-wrap gap-x-4 text-sm">
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#cb3300" width="15" height="15"></rect></svg><span>98.0–98.5%</span></div>
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#3163ce" width="15" height="15"></rect></svg><span>98.5–99.0%</span></div>
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#ff9b00" width="15" height="15"></rect></svg><span>99.0–99.5%</span></div>
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#02991d" width="15" height="15"></rect></svg><span>99.5–100.0%</span></div>
  </div>
  <img src="/assets/images/posts/advanced-horse-race-tactics/path_efficiency_surface_going.svg" alt="Win % by path efficiency vs. surface/going">
</figure>

> **Observation**
>
> Horses with a higher path efficiency generally have an advantage over horses with a lower path efficiency. This advantage is true across different distances and goings. Saving that extra 1% or 2% of the race distance compared to other competitors increases a horse’s win percentage.

#### Racing Strategy

Here, our Racing Strategy categories mirror the Running Style categories used in Brisnet’s Ultimate Past Performances[^brisnet]:

- **Early:** a horse that typically vies for the early lead.
- **Early Presser:** a horse that runs second or third within a few lengths of the lead early before trying to run down the leader.
- **Presser:** a horse that runs in the middle-of-the-pack early before trying to run down the leader.
- **Sustainer:** a horse that runs in the back of the pack early before trying to run down the leader.

<figure>
  <figcaption>Win % by racing strategy vs. distance</figcaption>
  <div class="mb-3 flex flex-row flex-wrap gap-x-4 text-sm">
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#cb3300" width="15" height="15"></rect></svg><span>Early</span></div>
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#3163ce" width="15" height="15"></rect></svg><span>Early Presser</span></div>
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#ff9b00" width="15" height="15"></rect></svg><span>Presser</span></div>
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#02991d" width="15" height="15"></rect></svg><span>Sustainer</span></div>
  </div>
  <img src="/assets/images/posts/advanced-horse-race-tactics/racing_strategy_distance.svg" alt="Win % by racing strategy vs. distance">
</figure>

<figure>
  <figcaption>Win % by racing strategy vs. surface/going</figcaption>
  <div class="mb-3 flex flex-row flex-wrap gap-x-4 text-sm">
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#cb3300" width="15" height="15"></rect></svg><span>Early</span></div>
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#3163ce" width="15" height="15"></rect></svg><span>Early Presser</span></div>
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#ff9b00" width="15" height="15"></rect></svg><span>Presser</span></div>
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#02991d" width="15" height="15"></rect></svg><span>Sustainer</span></div>
  </div>
  <img src="/assets/images/posts/advanced-horse-race-tactics/racing_strategy_surface_going.svg" alt="Win % by racing strategy percentage vs. surface/going">
</figure>

> **Observation**
>
> Once again, front-runners are favoured to win the race. However, it is challenging to implement a strategy to ensure that a horse leads 20 seconds into the race because many other horses are fighting for the front position. In addition, the advantage of being an early horse seems to decrease for longer distances and when the going is more muddy.

#### Speed Fluctuation

<figure>
  <figcaption>Win % by speed fluctuation vs. distance</figcaption>
  <div class="mb-3 flex flex-row flex-wrap gap-x-4 text-sm">
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#cb3300" width="15" height="15"></rect></svg><span>0.1–0.2</span></div>
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#3163ce" width="15" height="15"></rect></svg><span>0.2–0.3</span></div>
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#ff9b00" width="15" height="15"></rect></svg><span>0.3–0.4</span></div>
  </div>
  <img src="/assets/images/posts/advanced-horse-race-tactics/speed_fluctuation_distance.svg" alt="Win % by speed fluctuation vs. distance">
</figure>

<figure>
  <figcaption>Win % by speed fluctuation vs. surface/going</figcaption>
  <div class="mb-3 flex flex-row flex-wrap gap-x-4 text-sm">
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#cb3300" width="15" height="15"></rect></svg><span>0.1–0.2</span></div>
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#3163ce" width="15" height="15"></rect></svg><span>0.2–0.3</span></div>
    <div class="flex gap-x-2 items-center"><svg width="15" height="15"><rect fill="#ff9b00" width="15" height="15"></rect></svg><span>0.3–0.4</span></div>
  </div>
  <img src="/assets/images/posts/advanced-horse-race-tactics/speed_fluctuation_surface_going.svg" alt="Win % by speed fluctuation percentage vs. surface/going">
</figure>

> **Observation**
>
> Horses with a lower speed fluctuation perform significantly better across all different distances and race goings and is something that trainers and jockeys should prioritise. A lower speed fluctuation likely means that the horse has had a smooth race without being unexpectedly blocked or pushed by other horses.

### Predictive Power of the Features

To determine the predictive power of these four features, we can input them into a Multinomial Logit model and use the loss function to measure the performance of these features. One of the key advantages of this method is that it allows us to compare horses competing in the same race by creating performance ratings for each horse, before transforming them into win probabilities. We will compare the results with a baseline model where every horse has an equal probability of winning.

> **Multinomial Logit Explanation**
>
> The Multinomial Logit model was first applied to horse racing by Ruth Bolton and Randall Chapman[^bolton86] and further improved by William Benter[^benter08] who went on to amass a \$1bn fortune betting on horses in Hong Kong[^chellel18].
>
> The model works by weighting the features and adding them up to create a performance rating for each horse. The stronger the horse, the higher the performance rating. Afterwards, we group the performance ratings by race and use the softmax function to transform these performance ratings into probabilities. Finally, we use the Maximum Likelihood Estimation (MLE) technique to adjust the feature weights by maximising the log likelihood of the win probabilities of the horses that won.
>
> Mathematically, let $$X_{ijk}$$ represent the $$k$$-th feature of the $$j$$-th horse of the $$i$$-th race, $$\mathbf{1}_{ij}$$ represent an indicator function which returns 1 if the $$j$$-th horse is the winner of the $$i$$-th race and 0 otherwise, and $$\alpha_{k}$$ represent the weighting of the $$k$$-th feature. Then we have $$U_{ij}$$, which represents the performance rating of each horse:
>
> $$U_{ij} = \sum_{k}\alpha_{k}X_{ijk}$$
>
> To convert the performance rating to probabilities, we need to compare it with other horses in the same race. We group the performance ratings by race and perform the softmax function to convert them into win probabilities. Now, we have $$p_{ij}$$, which is the win probability of the $$j$$-th horse of the $$i$$-th race:
>
> $$p_{ij} = \frac{e^{U_{ij}}}{\sum_{j}e^{U_{ij}}}$$
>
> To get the maximum likelihood, we want to maximise:
>
> $$p = \prod_{i}p_{ij}\mathbf{1}_{ij}$$
>
> Which is the same as maximising the log likelihood function:
>
> $$L(p) = \ln\left(\prod_{i}p_{ij}\mathbf{1}_{ij}\right) = \sum_{i}\ln\left(p_{ij}\mathbf{1}_{ij}\right)$$

> **PyTorch Model**
>
> We turn to PyTorch to implement the Multinomial Logit model, since adjusting the feature weights to calculate the performance rating is the same as adjusting the weights of a single <code>nn.linear</code> layer neural network. PyTorch also comes with useful functions like <code>F.softmax</code>, which can be used to transform performance ratings to probabilities, and <code>F.nll_loss</code> which can be used to compare the predicted and target win probabilities.
>
> The data is loaded into a PyTorch Lightning data module which applies min-max normalisation to all non-odds features and pads races, since the number of starters in each race varies between 3 and 14. The data is divided into 5 folds for cross validation.
>
> A PyTorch Lightning trainer is used to find the ideal weights by training the neural network using an Adam optimiser with learning rate of 1e-3 and L2 weight decay of 1e-4, and letting it run for 200 epochs.

<figure class="tabular-nums overflow-x-auto" markdown="1">

| drafting_percentage | path_efficiency | racing_strategy | speed_fluctuation | split_0_val_loss | split_1_val_loss | split_2_val_loss | split_3_val_loss | split_4_val_loss | mean_val_loss |
| ------------------- | --------------- | --------------- | ----------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ------------- |
| ✓                   | ✓               | ✓               | ✓                 | 1.970482         | 1.947848         | 1.967863         | 1.961061         | 2.012110         | 1.971873      |
| ✓                   | ✓               | ✓               | ✗                 | 2.018924         | 1.989852         | 2.004551         | 2.009181         | 2.043011         | 2.013104      |
| ✓                   | ✓               | ✗               | ✓                 | 1.968063         | 1.979685         | 1.969283         | 1.997149         | 2.018979         | 1.986632      |
| ✓                   | ✓               | ✗               | ✗                 | 2.033243         | 2.019624         | 2.030846         | 2.023051         | 2.047879         | 2.030929      |
| ✓                   | ✗               | ✓               | ✓                 | 2.499842         | 2.504242         | 2.539999         | 2.512629         | 2.515285         | 2.514399      |
| ✓                   | ✗               | ✓               | ✗                 | 2.521984         | 2.548312         | 2.569168         | 2.551849         | 2.561134         | 2.550489      |
| ✓                   | ✗               | ✗               | ✓                 | 2.499354         | 2.503810         | 2.540120         | 2.512496         | 2.515368         | 2.514230      |
| ✓                   | ✗               | ✗               | ✗                 | 2.525560         | 2.548255         | 2.570994         | 2.553093         | 2.563932         | 2.552367      |
| ✗                   | ✓               | ✓               | ✓                 | 1.972460         | 1.957505         | 1.964547         | 1.977475         | 1.997277         | 1.973853      |
| ✗                   | ✓               | ✓               | ✗                 | 2.014825         | 1.992549         | 2.016295         | 2.008050         | 2.041398         | 2.014624      |
| ✗                   | ✓               | ✗               | ✓                 | 1.976256         | 1.970907         | 2.010788         | 1.988724         | 2.032701         | 1.995875      |
| ✗                   | ✓               | ✗               | ✗                 | 2.030576         | 2.019928         | 2.042306         | 2.043823         | 2.060228         | 2.039372      |
| ✗                   | ✗               | ✓               | ✓                 | 2.530894         | 2.530661         | 2.556012         | 2.538270         | 2.527853         | 2.536738      |
| ✗                   | ✗               | ✓               | ✗                 | 2.570032         | 2.594265         | 2.600477         | 2.593902         | 2.590201         | 2.589775      |
| ✗                   | ✗               | ✗               | ✓                 | 2.549052         | 2.535519         | 2.559433         | 2.543592         | 2.532526         | 2.544024      |
| ✗                   | ✗               | ✗               | ✗                 | 2.639057         | 2.639057         | 2.639057         | 2.639057         | 2.639057         | 2.639057      |

</figure>

> **Observation**
>
> The predictive power using our four features is significantly better than the baseline model with the baseline giving a loss of 2.639 and our combined model giving us a loss of 1.965. We can analyse the weightings of the features to help inform racing tactics and decisions.
>
> The feature weights inside the combined model, indicate the following, in order of significance:
>
> 1. A higher path efficiency gives a very significant advantage (mean weight of 7.702)
> 2. A lower speed fluctuation gives a moderate advantage (mean weight of -3.609)
> 3. An early runner racing strategy gives a slight advantage (mean weight of -0.559)
> 4. Drafting percentage does not make much difference (mean weight of -0.081)

### Are the odds efficient?

The probability implied by the odds is supposed to factor in all facets of the race including drafting benefit, path efficiency, racing strategy and speed fluctuation. However, this may only sometimes be the case because there might be inefficiencies in the odds. In this section, we will use the Multinomial Logit model again to analyse whether the odds are efficient using only the starter-specific features which are available before the race, excluding features derived from coordinate data. The only starter-specific feature in the NYTHA/NYRA dataset is `weight_carried`, but we supplement this with `draw`, which we calculate using each starter’s distance to the centroid at the first Trakus measurement.

<figure class="tabular-nums overflow-x-auto" markdown="1">

| public_odds | draw | weight_carried | split_0_val_loss | split_1_val_loss | split_2_val_loss | split_3_val_loss | split_4_val_loss | mean_val_loss |
| ----------- | ---- | -------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ------------- |
| ✓           | ✓    | ✓              | 1.633186         | 1.615052         | 1.602511         | 1.643421         | 1.593186         | 1.617471      |
| ✓           | ✓    | ✗              | 1.633929         | 1.615895         | 1.603535         | 1.643102         | 1.591764         | 1.617645      |
| ✓           | ✗    | ✓              | 1.633831         | 1.614715         | 1.602664         | 1.644175         | 1.593298         | 1.617737      |
| ✓           | ✗    | ✗              | 1.634561         | 1.615534         | 1.603706         | 1.643845         | 1.591892         | 1.617908      |

</figure>

> **Observation**
>
> The mean loss for both models are very similar which means that adding the two extra features of <code>draw</code> and <code>weight_carried</code> did not improve the odds probabilities significantly. The public odds have already factored in the two extra features which means that the odds are already efficient enough to reflect all the starter-specific pre-race information provided in the NYTHA/NYRA dataset.

### Do our features improve the odds?

We will now add the four newly generated features to the odds’ implied utility to see whether the model’s predictive power improves. It is worth noting that we do not have these features available before the race because they are generated from information during the race, but we may be able to forecast these features using a horse’s past performances.

<figure class="tabular-nums overflow-x-auto" markdown="1">

| public_odds | drafting_percentage | path_efficiency | racing_strategy | speed_fluctuation | split_0_val_loss | split_1_val_loss | split_2_val_loss | split_3_val_loss | split_4_val_loss | mean_val_loss |
| ----------- | ------------------- | --------------- | --------------- | ----------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ------------- |
| ✓           | ✓                   | ✓               | ✓               | ✓                 | 1.495163         | 1.459962         | 1.433159         | 1.487816         | 1.450319         | 1.465284      |
| ✓           | ✓                   | ✗               | ✗               | ✗                 | 1.612913         | 1.574613         | 1.560399         | 1.595606         | 1.543536         | 1.577413      |
| ✓           | ✗                   | ✓               | ✗               | ✗                 | 1.633423         | 1.610808         | 1.597784         | 1.636444         | 1.592358         | 1.614163      |
| ✓           | ✗                   | ✗               | ✓               | ✗                 | 1.622015         | 1.581996         | 1.576854         | 1.614290         | 1.566291         | 1.592289      |
| ✓           | ✗                   | ✗               | ✗               | ✓                 | 1.528886         | 1.512381         | 1.491406         | 1.547286         | 1.505368         | 1.517065      |
| ✓           | ✗                   | ✗               | ✗               | ✗                 | 1.634561         | 1.615534         | 1.603706         | 1.643845         | 1.591892         | 1.617908      |

</figure>

> **Observation**
>
> When we add all four features to the odds, there is a significant improvement in accuracy, dropping from 1.618 to 1.468. If we can forecast these features with reasonable accuracy, we can produce more accurate probabilities than the ones implied by the odds. Out of the four features we added, the one that helps improve the model the most is the speed fluctuation feature. This means that public odds largely undervalues the performance benefits of reduced speed fluctuation.
>
> The feature weights inside the combined model, indicate the following, in order of significance:
>
> 1. A lower speed fluctuation gives a very significant advantage over public odds (mean weight of -8.810)
> 2. A lower drafting percentage gives a moderate advantage over public odds (mean weight of -1.135)
> 3. An early runner racing strategy gives a slight advantage over public odds (mean weight of -0.450)
> 4. A lower path efficiency gives a slight advantage over public odds (mean weight of -0.433)

## Conclusion

> **Key Takeaways**
>
> - Jockeys and trainers should choose race tactics that prioritise **increasing path efficiency** and **decreasing speed fluctuation** over drafting percentage and racing strategy.
> - **Public odds significantly undervalue horses with lower speed fluctuations.** If these undervalued horses can be identified before the race, this inefficiency can be exploited by placing exotic bets covering these horses.

### Recommended Race Tactics

Jockeys and trainers should prioritise increasing path efficiency and decreasing speed fluctuation in their race tactics because they both significantly increase the probability of winning a race. However, these two things may sometimes conflict with each other because the shortest path is close to the rail where it might be congested with other horses which might increase speed fluctuation. Hence, it is important find a balance between two and perhaps find a strategy which can maximise both path efficiency and speed fluctuation.

### Recommended Betting Strategies

Public odds significantly undervalue horses with lower speed fluctuations. If these undervalued horses can be identified before the race, this inefficiency can be exploited by placing exotic bets covering these horses. By betting on multiple value horses at the same time, we have more opportunities to overcome the takeout rate. Bet sizes should be placed in accordance with the Kelly Criterion[^kelly56], in order to maximise the expected growth rate of wealth.

## References

[^brisnet]: [How To Read Brisnet.com Ultimate Past Performances. (n.d.). _Brisnet.com_.](http://www.brisnet.com/content/brisnet-online-horse-racing-data-handicapping/read-brisnet-com-ultimate-past-performances)

[^benter08]: [Benter, W. (2008). Computer Based Horse Race Handicapping and Wagering Systems: A Report. _Efficiency of Racetrack Betting Markets_.](https://www.gwern.net/docs/statistics/decision/1994-benter.pdf)

[^bolton86]: [Bolton, R. N., & Chapman, R. G. (1986). Searching for Positive Returns at the Track: A Multinomial Logit Model for Handicapping Horse Races. _Management Science_.](https://www.gwern.net/docs/statistics/decision/1986-bolton.pdf)

[^chellel18]: [Chellel, K. (2018). The Gambler Who Cracked the Horse-Racing Code. _Bloomberg Businessweek._](https://www.bloomberg.com/news/features/2018-05-03/the-gambler-who-cracked-the-horse-racing-code)

[^kelly56]: [Kelly, J. L. (1956). A New Interpretation of Information Rate. _Bell System Technical Journal_.](https://www.princeton.edu/~wbialek/rome/refs/kelly_56.pdf)
