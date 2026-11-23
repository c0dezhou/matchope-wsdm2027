"""Matching value for two users who prefer different items."""

import numpy as np

from matchope.eval.matching_value import marginalize_policy, matching_value


policy = np.array([[1.0, 0.0], [0.0, 1.0]])
outcomes = np.array([[1.0, 0.0], [0.0, 1.0]])
shared_contexts = np.array([0, 0])
reference = marginalize_policy(policy, shared_contexts)

print(f"Target value: {np.mean(np.sum(policy * outcomes, axis=1)):.1f}")
print(f"Reference value: {np.mean(np.sum(reference * outcomes, axis=1)):.1f}")
print(f"Matching value: {matching_value(policy, outcomes, shared_contexts):.1f}")
