# Soccer RNN

A Recurrent Neural Net trained on soccer goals. Trained to predict the result, given only the scoreline, each 10 minutes throughout the match. All premised on the idea that there is some predictive power in when in a match one team scores on another. Does a pretty okay job, given that it has no information about the teams other than whether they're at home or away, and what time they've scored in the match up to that point. By half time it gets about 75% of the results correct which is ~10% better than you would do by just picking the team that's ahead at half time.

![preds](https://github.com/neal-o-r/soccernn/blob/master/match.png)
