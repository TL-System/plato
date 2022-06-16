# Region assignment environment

Social networking applications need to make decisions about where to geolocate data. If data is located near the user requesting it, access latency improves. If data is located far away from the user requesting it, access latency worsens. Certain pages are more likely to be viewed by users in a particular region (e.g., Chinese pages are more likely to be viewed in China). Generally, human experts must craft heuristics to predict the ideal region to locate a piece of data (e.g., place each piece of data in the same region as the data's creator), or must relocate data after many out-of-region accesses have occurred.

This task represents replacing such systems with a predictive one: when a new piece of data (a page) is created, a model must decide where to place it based on a few features of the created data.

## Inputs

Each state represents a region assignment for a single page. Each state includes:

* The region the page was created in
* An estimate of the language used on the page
* Whether or not the account posting the page has linked to each of 100 popular websites (e.g. Twitter)

## Outputs

The output at each step should be an assignment of the observed page to one of 8 regions. A page can only be assigned to one region.

## Reward

The agent will receive a reward equal to the amount of in-region traffic minus the amount of out-of-region traffic the chosen assignment entails. This is likely to be negative, as one region rarely represents a majority of traffic to a page (e.g., 45% of a page's views may come from one region, but the other 7 regions account for 7.8% each).

## Configuration options

By default, pages will be created in a random order. Setting the `ra_shuffle` option to false will cause the pages to be presented to the agent in the same order that they were created in the recorded system.
