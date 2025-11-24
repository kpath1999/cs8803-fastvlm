# Cross-Season Matching Performance Summary

| Method | Total Queries | Total Matches | Matches/Query | Top-1 Quality | Avg Quality | High Quality % | Label Jaccard | Avg IoU | Match Rate | Visual Score | Semantic Score | Geometric Score | Avg Inliers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline A (Visual+Semantic) | 10 | 55 | 5.500 | 0.500 | 0.479 | 38.182 | 0.836 | 0.252 | 0.179 | 0.874 | 0.836 | N/A | N/A |
| Baseline B (Geometric) | 10 | 50 | 5.000 | 0.356 | 0.345 | 4.000 | 0.559 | 0.191 | 0.090 | N/A | N/A | 1.000 | 4.020 |
| Full Pipeline (Multi-Modal) | 10 | 50 | 5.000 | 0.482 | 0.461 | 28.000 | 0.806 | 0.242 | 0.165 | 0.869 | 0.806 | 0.900 | 3.760 |


## Metric Definitions

- **Total Queries**: Number of unique query frames tested
- **Total Matches**: Number of candidate matches returned (top-5 per query)
- **Matches/Query**: Average number of matches per query
- **Top-1 Quality**: Average quality score of the best match
- **Avg Quality**: Average quality score across all matches
- **High Quality %**: Percentage of matches with quality > 0.5
- **Label Jaccard**: Average Jaccard similarity of semantic labels
- **Avg IoU**: Average intersection-over-union of bounding boxes
- **Match Rate**: Average percentage of detected objects successfully matched
- **Visual/Semantic/Geometric Score**: Component scores from each modality
- **Avg Inliers**: Average number of RANSAC inliers for geometric verification
