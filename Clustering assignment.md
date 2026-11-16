# Gym member segments: who should get the monthly offer?

## The question

A gym chain sends every member the same monthly email, with the same offer. **Does the member base fall into distinct groups such that the gym could target the offer, and how much is that worth compared to sending everyone the same thing?**

## The model

**Clustering (k-means).** There is no outcome column to predict: the model proposes groups of similar members from their features. Use python to fit via k-means and report cluster assignments and centers.

## Context

The operations lead wants three to five member segments and a one-line description of each. She also wants to know what segment a new member falls into after their first month, so the right offer starts early.

Today's monthly email includes a **$10 extras credit** (smoothie bar, merchandise, or personal training) for every member. Finance's working assumptions:

- Every member who receives the credit uses it, so it costs the gym $10 per member per month.
- Members who already spend **$20 or more** a month on extras would spend the same without the credit. For them, the credit is a $10 giveaway.
- Members who spend **less than $20** a month on extras spend $25 more when they receive the credit. At a 60% margin on extras, that is $15 of margin, so the credit nets the gym +$5 for each of them.

The **baseline** is a single group: every member is treated as the same average member, which is the current plan of one email and the credit to everyone.

## The data

**`gym_members.csv`**: 330 members, one row per member, averages over the last 3 months.

| Column | Meaning |
|---|---|
| `member_id` | member identifier |
| `age` | member age |
| `visits_per_month` | gym visits per month |
| `avg_visit_minutes` | average minutes per visit |
| `classes_per_month` | group classes attended per month |
| `months_member` | months since joining |
| `monthly_extras_spend` | dollars per month spent beyond the base membership |

Data checks: 330 rows, 7 columns; average extras spend about $26 a month.

## The task

1. **Split the data.** Randomly set aside 30% of the 330 members as a test set. Fit everything, including the scaling, on the remaining 70% (the training set). The test members are only for evaluating fit.
2. **Fit the model.** Choose the features that describe how a member uses the gym and who they are, and leave out any column that does not. Fit k-means for a range of group counts (k = 1 to 8) on the training members, plot the within-group spread (the sum of squared distances from each member to its group's center) against k, and choose k from the elbow. The single-group model (k = 1) is the baseline. There are no actual groups to compare against, so the comparison is between candidate models that differ only in k. Describe each of your chosen k-model groups based on its centroid (average member) and your interpretation of what their characteresitics say about them, and then assign them a name. For example, a single-expensive-item and infrequent purchaser on an ecommerce platorm might be a "splurger".
3. **Evaluate fit on the test members.** Do not refit. For each k, take the group centers fitted on the training members, assign each test member to its nearest center, and sum the squared distances. Plot this test spread against k next to the training curve. Confirm the elbow is at the same k, and compare the spread at k = 1 (baseline) to the spread at your k on the test members. If the test curve has no elbow at your k, the groups did not generalize; say so and reconsider k. The comparison is **baseline vs v1 on members the model never saw**.
4. **Make the decision.** Decide which segments get the $10 credit. Write a function that takes one member's features and returns the segment and whether it gets the credit. Apply it to a new member: age 30, 4 visits a month, 42 minutes per visit, 1 class a month, 5 months a member, $10 a month on extras.
5. **Quantify it.** Assign every member (training and test) to a segment with the fitted model. Using finance's assumptions, compute the gym's monthly net from the credit under the baseline (credit to everyone) and under v1 (credit only to the segments you chose). Apply the assumptions member by member, not to segment averages.
6. **Present it.** Make two slides for the operations lead, not for a statistics audience.

## Deliverables
Note: you may use AI throughout and it is expected that you do so for coding, however, do not use AI to generate your slides. The value of this assignment is interpreting the technical output and synthesizing for business decision-makers. It is better to have succinct black text on white background that you wrote yourself rather than volumes of AI-generated nonsense for your "boss" to have to parse. (Asking AI to write python code to create graphs/charts for your slides is expected, copying-pasting an a graph made by ChatGPT/Claude/Gemini is not.)

One submission: **one PDF.**

1. **Title slide:** your name and the link to your GitHub repo containing the code and analysis for the project. The code reproduces every number on your slides.
2. **Two slides covering:**
   - Data: source, rows, period, the 70/30 split.
   - Model specification: features used and any left out, scaling, model family, the k you chose.
   - Fit baseline vs v1 comparisons: the elbow plot with the training and test curves, the spread at k = 1 and at your k on the test members, and one line on why you chose k and that the test curve agrees.
   - Business decision and value quantification:
   -- Headline in one sentence- what targeting changes and what it is worth
   -- A table with one column per segment: member count, two or three defining traits, extras spend, and whether it gets the credit; the new member's segment
   -- Compare value quantification of baseline vs v1: the monthly net from the credit under each plan
