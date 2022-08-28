import time
import sys
import numpy as np
import random
import argparse
import pandas as pd
from functools import reduce
import math
from timeit import default_timer as timer

LON_MIN = 5481
LON_MAX = 1283272858
LAT_MIN = 1060738932
LAT_MAX = 1717227584
TS_MIN = 1132191726
TS_MAX = 1552511621
BUILDING_IX = 7
HIGHWAY_IX = 11
NODE_IX = 1
WAY_IX = 3
PINF = 1 << 50
NINF = -(1 << 50)
NUM_QUERY_TYPES = 7

## Business questions for this query:
# How many elements were added by users in a time window of length N? (1)
# How many elements exist in a particular lat/lon box? (2)
# How many elements were added by users in a particular lat/lon box in the last N years? (3)
# How many buildings are there in a lat/lon region? (4)
# How many highways are there in a lat/lon region? (5)
# How many ways were added in the last N year?
# How many nodes were added in the last N years? (7)
# How many buildings were added in the last N years in a particular lat/lon box? (unused)

def set_bounds(data):
    global LON_MIN, LON_MAX, LAT_MIN, LAT_MAX
    global TS_MIN, TS_MAX
    LON_MIN = np.min(data[data[:,1] > 0, 1])
    LON_MAX = np.max(data[: 1])
    LAT_MIN = np.min(data[data[:,2] > 0, 2])
    LAT_MAX = np.max(data[:, 2])
    TS_MIN = np.min(data[:, 3])
    TS_MAX = np.max(data[:, 3])

class CDFHist:
    def __init__(self, data, buckets):
        # Instead of a normal histogram, makes one where, if there are k buckets, each holds 1/kth
        # the points.
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        if len(buckets) > 1:
            data = data[np.argsort(data[:,1]), :]
            max2 = np.max(data[:, 1])
            ixs = (np.linspace(0, 1.0, buckets[1]+1) * len(data)).astype(int)
            f2_vals = np.append(data[ixs[:-1], 1], max2+1)
        if len(buckets) > 2:
            data = data[np.argsort(data[:,2]), :]
            max3 = np.max(data[:, 2])
            ixs = (np.linspace(0, 1.0, buckets[2]+1) * len(data)).astype(int)
            f3_vals = np.append(data[ixs[:-1], 2], max3+1)
        data = data[np.argsort(data[:,0]), :]
        max0 = np.max(data[:, 0])
        ixs = (np.linspace(0, 1.0, buckets[0]+1) * len(data)).astype(int)
        f1_vals = np.append(data[ixs[:-1], 0], max0+1)
        
        f2_cnts = []
        f3_cnts = []
        if len(buckets) > 1:
            for b in range(buckets[0]):
                arr = data[ixs[b]:ixs[b+1], :]
                arr = arr[np.argsort(arr[:,1]),:]
                ixs2 = np.searchsorted(arr[:,1], f2_vals, side='left')
                f2_cnts.append((ixs2.astype(float)))

                if len(buckets) > 2:
                    f3_cnts.append([])
                    assert len(f3_cnts) == b+1
                    for c in range(buckets[1]):
                        arr3 = arr[ixs2[c]:ixs2[c+1], :]
                        if len(arr3) == 0:
                            f3_cnts[b].append(np.zeros(len(f3_vals)))
                            continue
                        arr3 = arr3[np.argsort(arr3[:,2]), :]
                        ixs3 = np.searchsorted(arr3[:,2], f3_vals, side='left')
                        f3_cnts[b].append((ixs3.astype(float)))

        if len(buckets) > 1:
            self.dim2_vals = f2_vals
            self.dim2_counts = np.array(f2_cnts)
        if len(buckets) > 2:
            self.dim3_vals = f3_vals
            self.dim3_counts = np.array(f3_cnts)
        self.dim1_vals = f1_vals 
        self.buckets = buckets

    # Given a cdf value (between 0 and 1), finds the data value corresponding to it.
    # If not querying the first dimension in the histogram, provide the CDF ranges
    # of the higher dimensions.
    def value_for_cdf(self, cdf, buckets, arr):
        first = int(cdf * buckets)
        second = first+1
        first_cdf = float(first)/buckets
        second_cdf = float(second)/buckets
        if len(arr) == 0:
            return None
        return ((cdf - first_cdf)/(second_cdf - first_cdf)) * (arr[second] - arr[first]) + arr[first]

    def values_for_cdf1(self, cdf_start, cdf_end):
        vals = np.interp([cdf_start, cdf_end], np.arange(self.buckets[0]+1)/self.buckets[0], self.dim1_vals) 
        ranges = np.searchsorted(np.arange(self.buckets[0]+1)/self.buckets[0], [cdf_start, cdf_end])
        return vals, np.maximum(0, ranges-1)

    def values_for_cdf2(self, cdf_start, cdf_end, range1, side='bottom'):
        # Normalize to get actual cumulative sums
        cumuls = np.sum(self.dim2_counts[range1[0]:range1[1]+1,:], axis=0)
        cumuls /= np.max(cumuls)
        assert np.all(cumuls <= 1.0)
        vals = np.interp([cdf_start, cdf_end], cumuls, self.dim2_vals)
        ranges = np.searchsorted(cumuls, [cdf_start, cdf_end])
        return vals, np.maximum(0, ranges-1)

    def values_for_cdf3(self, cdf_start, cdf_end, range1, range2, side='bottom'):
        # Normalize to get actual cumulative sums
        cumuls = np.sum(self.dim3_counts[range1[0]:range1[1], range2[0]:range2[1], :], axis=(0,1))
        cumuls /= np.max(cumuls)
        return np.interp([cdf_start, cdf_end], cumuls, self.dim3_vals)
        
    
class QueryGen:
    def __init__(self, datafile, sample=10000000):
        self.rand = np.random.RandomState(0)
        dataset = np.fromfile(datafile, dtype=np.int64).reshape(-1, 6)
        ixs = self.rand.choice(len(dataset), sample, replace=False)
        data = dataset[ixs, :]
        self.data = data[np.argsort(data[:,3]), :]

        # Choose a selectivity between 5e-5 and 5e-2. However, we want to choose selectivities
        # uniformly over a *logarithmic* space, so each power of 10 is chosen with equal prob.
        self.target_selectivity = 5 * math.pow(10, (self.rand.random_sample() * 3 - 5))
        print('Target selectivity =', self.target_selectivity)
        
        self.hist, self.edges = None, None
        self.hist_clean, self.edges_clean, self.ixs_clean = None, None, None
        self.hist_bld, self.edges_bld, self.ixs_bld = None, None, None
        self.hist_hwy, self.edges_hwy, self.ixs_hwy = None, None, None
        self.hist_way, self.edges_way, self.ixs_way = None, None, None
        self.gen_histogram()
        # Generate a probability distribution over the query types.
        self.query_probs = np.cumsum(self.rand.random_sample(NUM_QUERY_TYPES))
        self.query_probs /= np.max(self.query_probs)

    def seed(self, seed):
        self.rand.seed(seed)

    def gen_histogram(self):
        #self.data = self.data[(self.data[:,2] > 0) + (self.data[:,3] > 0), 2:5]
        self.ixs_clean = np.where((self.data[:,1] > 0) * (self.data[:,2] > 0))[0]
        self.ixs_bld = np.where((self.data[:,4] == WAY_IX) * (self.data[:,5] == \
            BUILDING_IX))[0]
        self.ixs_hwy = np.where((self.data[:,4] == WAY_IX) * (self.data[:,5] == \
            HIGHWAY_IX))[0]
        self.ixs_way = np.where((self.data[:,4] == WAY_IX))[0]

        data_cols = self.data[:, np.array([3, 1, 2])]
        self.hist = CDFHist(self.data[:, 3], (10000,))
        self.hist_clean = CDFHist(data_cols[self.ixs_clean, :], (800, 800, 300))
        self.hist_bld = CDFHist(data_cols[self.ixs_bld, :], (500, 300, 100))
        self.hist_hwy = CDFHist(data_cols[self.ixs_hwy, :], (500, 300, 100))
        self.hist_way = CDFHist(self.data[self.ixs_way, 3], (10000,))

    def range_impl(self, sels, hist):
        # All in the time dimension
        vals1 = (NINF, PINF)
        vals2 = (NINF, PINF)
        vals3 = (NINF, PINF)
        ranges1 = (0, hist.buckets[0])
        if sels[2] < 1:
            sel = sels[2]
            start1 = self.rand.random_sample() * (1 - sel)
            end1 = start1 + sel
            vals1, ranges1 = hist.values_for_cdf1(start1, end1)
        if sels[0] < 1:
            ranges2 = (0, hist.buckets[1])
            sel = sels[0] 
            start2 = self.rand.random_sample() * (1-sel)
            end2 = start2 + sel
            vals2, ranges2 = hist.values_for_cdf2(start2, end2, ranges1)
            if sels[1] < 1:
                sel = sels[1]
                start3 = self.rand.random_sample() * (1-sel)
                end3 = start3 + sel
                vals3 = hist.values_for_cdf3(start3, end3, ranges1, ranges2)
        return vals2, vals3, vals1

    
    def range(self, target_sels, bld=False, hwy=False, clean=False, way=False):
        r = None
        count = 0
        while r is None:
            count += 1
            if clean:
                r = self.range_impl(target_sels, self.hist_clean)
            elif way:
                r = self.range_impl(target_sels, self.hist_way)
            elif bld:
                r = self.range_impl(target_sels, self.hist_bld)
            elif hwy:
                r = self.range_impl(target_sels, self.hist_hwy)
            else:
                r = self.range_impl(target_sels, self.hist)
        
        return r

    def random_query(self):
        start = [NINF] * 6
        end = [PINF] * 6

        target_sel = self.target_selectivity
        choice = self.rand.random_sample()
        if choice < self.query_probs[0]:
            _, _, s = self.range([1, 1, target_sel])
            start[3], end[3] = s[0], s[1]
        elif choice < self.query_probs[1]:
            targ = math.sqrt(target_sel)
            var = 0.1 * targ
            r = self.rand.random_sample() * var + (targ - var/2)
            a, b, _ = self.range([r, target_sel/r, 1], clean=True)
            start[1], end[1] = a[0], a[1]
            start[2], end[2] = b[0], b[1]
        elif choice < self.query_probs[2]:
            r = self.rand.random_sample() * 0.05 + target_sel #0.4 + 0.2
            targ = math.sqrt(target_sel/r)
            var = 0.1 * targ
            w = self.rand.random_sample() * var + (targ - var/2)
            a, b, c = self.range([w, target_sel/(w*r), r], clean=True)
            start[1], end[1] = a[0], a[1]
            start[2], end[2] = b[0], b[1]
            start[3], end[3] = c[0], c[1]
        elif choice < self.query_probs[3]:
            new_target = (target_sel * len(self.data)) / len(self.ixs_bld)
            _, _, s = self.range([1, 1, new_target], bld=True)
            start[3], end[3] = s[0], s[1]
            start[4], end[4] = WAY_IX, WAY_IX
            start[5], end[5] = BUILDING_IX, BUILDING_IX
        elif choice < self.query_probs[4]:
            new_target = (target_sel * len(self.data)) / len(self.ixs_hwy)
            _, _, s = self.range([1, 1, new_target], hwy=True)
            start[3], end[3] = s[0], s[1]
            start[4], end[4] = WAY_IX, WAY_IX
            start[5], end[5] = HIGHWAY_IX, HIGHWAY_IX
        elif choice < self.query_probs[5]:
            new_target = (target_sel * len(self.data)) / len(self.ixs_way)
            _, _, s = self.range([1, 1, new_target], way=True)
            start[3], end[3] = s[0], s[1]
            start[4], end[4] = WAY_IX, WAY_IX
        elif choice < self.query_probs[6]:
            new_target = (target_sel * len(self.data)) / len(self.ixs_clean)
            _, _, s = self.range([1, 1, new_target],clean=True)
            start[3], end[3] = s[0], s[1]
            start[4], end[4] = NODE_IX, NODE_IX
        else:
            new_target = (target_sel * len(self.data)) / len(self.ixs_bld)
            r = self.rand.random_sample() * 0.4 + 0.2
            targ = math.sqrt(new_target/r)
            var = 0.1 * targ
            w = self.rand.random_sample() * var + (targ - var/2)
            a, b, c = self.range([w, new_target/(w*r), r], bld=True)
            start[1], end[1] = a[0], a[1]
            start[2], end[2] = b[0], b[1]
            start[3], end[3] = c[0], c[1]
            start[5], end[5] = BUILDING_IX, BUILDING_IX
             
        return start, end

