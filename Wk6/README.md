Week 6, starting on 2 August 2021 <br> <br>
Note: Run code from MAIN6.ipynb

# Schedule
* Mon: Weekly update. 
  * Update congestion algorithm. 
  * Try "random" dataset / merge "random" & "targeted" datasets for Blenheim clustering - taking into account congestion
* Tue: Generalise between similar edges (test using KS / CRPS)
  * Try using one edge in the cluster to fit all other edges
  * Try using a few datapoints from all edges in cluster to fit all edges in cluster
* Wed: Generalise between dissimilar edges (test using KS / CRPS)
  * Is there a mapping between lognormal model parameters based on edge length?
  * Do end nodes matter at all?
  * What is the difference between robots that are starting froms stationary, stopping, turning, moving straight?
* Thu: What maps would be interesting for generalisation?
  * Check STRANDS maps & data
  * What building blocks would be useful? E.g. random data collection between arbitrarily spaced nodes on an edge
* Fri: How many datapoints are needed to build a good representation of an edge? 
  * Does this change if you use datapoints from similar/dissimilar edges instead of the true edge?
  * Is it beneficial to ensure that more weight is placed on data from the true edge?
