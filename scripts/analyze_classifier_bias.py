#!/usr/bin/env python3
"""
analyze_classifier_bias.py - Analyze classifier weight distribution

This script checks if the classifier weights have a bias that causes
the model to always predict one class.
"""

import numpy as np

# Classifier weights from weights.h (extracted)
classifier_weight = np.array([
    # Row 0 (NEG class)
    -0.00659447, -0.02511095, 0.01547954, -0.03214066, 0.03002011, 0.00525298, 0.03815036, -0.03149528,
    -0.00644415, 0.00895839, 0.02128461, -0.00375015, -0.05414563, -0.02214189, 0.00361849, 0.03716904,
    -0.03803093, 0.00112481, -0.00291191, 0.00185953, 0.02299339, -0.00833588, -0.05159353, -0.02932330,
    -0.01136178, 0.03713692, 0.02161053, -0.01261431, -0.00011158, -0.01772799, -0.01666172, -0.01148918,
    -0.05508139, -0.03654043, 0.05094714, -0.01888202, 0.02690505, -0.03029699, -0.00238350, 0.00656701,
    0.00683493, 0.04537330, 0.00554513, -0.00569787, 0.03368826, -0.02449184, 0.03586372, -0.01714926,
    -0.00749927, 0.03098936, 0.02624647, 0.00360459, -0.01422609, -0.03841012, -0.02295213, 0.02997808,
    0.03490409, -0.03307981, -0.00883124, 0.01930331, 0.00501144, 0.04364957, 0.01583325, -0.03922465,
    0.00515608, -0.03788608, 0.02210715, 0.02683676, 0.00827631, 0.03477012, -0.04918249, 0.00252239,
    0.04053928, 0.01628575, 0.05135407, -0.00402025, -0.02866400, 0.03376157, -0.00106572, -0.00305775,
    -0.00427431, -0.00188194, 0.01054938, 0.04235762, 0.02200786, -0.02306566, 0.00460006, 0.00751405,
    -0.00165177, -0.02507683, 0.00320259, -0.00713757, 0.00938106, -0.01422264, -0.00077028, 0.04897146,
    0.00160446, -0.02510608, 0.01356431, -0.00528400, 0.04535211, -0.02375099, 0.02622071, -0.01595893,
    0.06352200, 0.01132051, -0.01080313, -0.00274177, -0.01252191, 0.01958925, -0.01931104, 0.00486925,
    -0.00585061, 0.01207492, 0.00592362, 0.01445431, -0.00767409, -0.01188934, 0.03360955, 0.04623865,
    0.01319745, -0.05273818, -0.05327497, -0.00435358, 0.01903108, 0.00701549, 0.02586765, -0.04988818,
    # Row 1 (POS class)
    0.03941969, 0.04740673, 0.00072701, 0.03305033, 0.02843008, 0.00420495, -0.03499689, -0.00051372,
    0.00676007, -0.02292646, -0.02933273, 0.03775986, 0.01302416, 0.00060032, 0.00753893, -0.03488967,
    0.02473587, 0.00123206, -0.03763635, 0.01347614, 0.01661755, 0.00648604, 0.00072830, -0.00286709,
    -0.00599683, -0.00133514, -0.02018205, 0.01646712, 0.02010764, -0.00135516, 0.00609408, -0.01778931,
    -0.00444939, 0.00022707, -0.01506310, -0.02177798, -0.04554801, 0.02681924, -0.00175342, 0.00140496,
    0.02231647, -0.00862460, 0.04328820, 0.02203102, -0.01117251, 0.00085411, -0.01944377, 0.04140390,
    -0.02475473, -0.01140016, -0.01548583, -0.00286309, 0.03371960, 0.01613372, 0.00529323, -0.02791544,
    0.02120079, 0.04227170, -0.02610859, -0.02347104, 0.02179774, -0.03263244, -0.03311088, 0.01362444,
    0.01009935, -0.00072549, -0.03547184, -0.01327145, -0.01027419, 0.01391506, 0.00786979, -0.02706994,
    -0.01989721, 0.01348650, -0.02429326, -0.00917680, -0.01780145, -0.01236325, -0.00265199, 0.02526874,
    0.01671640, 0.04427488, -0.01908852, 0.01064555, -0.02686686, 0.01664827, -0.07165919, 0.00840134,
    0.06286915, 0.02581616, -0.01230299, 0.02586042, -0.05231786, -0.01284773, -0.00230087, 0.00474089,
    0.03841716, 0.01124191, -0.00283189, -0.00130393, 0.01178070, 0.01283199, -0.01941384, 0.01605459,
    -0.03121025, -0.01337045, 0.03571676, 0.04351896, -0.01511430, -0.06141632, 0.03456497, -0.02343300,
    -0.00517463, -0.02210277, 0.00650156, -0.06197874, -0.01384402, 0.02494067, -0.04023191, -0.00898510,
    -0.03298094, -0.00593296, 0.03409729, -0.00221860, -0.01010833, -0.00046563, -0.01466783, -0.02236076,
]).reshape(2, 128)

classifier_bias = np.array([-0.01949221, 0.01949221])

print("=" * 60)
print("Classifier Weight Analysis")
print("=" * 60)

# Row statistics
print("\n1. Row Statistics:")
for i, label in enumerate(["NEG (row 0)", "POS (row 1)"]):
    row = classifier_weight[i]
    print(f"  {label}:")
    print(f"    Mean:   {row.mean():.6f}")
    print(f"    Std:    {row.std():.6f}")
    print(f"    Sum:    {row.sum():.6f}")
    print(f"    Min:    {row.min():.6f}")
    print(f"    Max:    {row.max():.6f}")

# Difference between rows
print("\n2. Row Difference (POS - NEG):")
diff = classifier_weight[1] - classifier_weight[0]
print(f"  Mean diff:   {diff.mean():.6f}")
print(f"  Sum diff:    {diff.sum():.6f}")

# Simulate with typical pooled values
print("\n3. Simulated Logits with uniform pooled=0.5:")
pooled_uniform = np.full(128, 0.5)
logits_uniform = classifier_weight @ pooled_uniform + classifier_bias
print(f"  Logits: NEG={logits_uniform[0]:.4f}, POS={logits_uniform[1]:.4f}")
print(f"  Prediction: {'POS' if logits_uniform[1] > logits_uniform[0] else 'NEG'}")

print("\n4. Simulated Logits with typical tanh values from debug:")
# From debug output: 0.890, 0.790, -0.712, 0.801, -0.132, -0.365, -0.957, 0.878...
# Let's assume a mean around 0.2 (slightly positive)
pooled_positive = np.random.uniform(-0.5, 1.0, 128)  # slightly positive bias
pooled_positive = pooled_positive / np.abs(pooled_positive).max()  # normalize to [-1, 1]
pooled_positive = pooled_positive * 0.8 + 0.1  # shift to have mean ~0.1

logits_positive = classifier_weight @ pooled_positive + classifier_bias
print(f"  Pooled mean: {pooled_positive.mean():.4f}")
print(f"  Logits: NEG={logits_positive[0]:.4f}, POS={logits_positive[1]:.4f}")
print(f"  Prediction: {'POS' if logits_positive[1] > logits_positive[0] else 'NEG'}")

print("\n5. What pooled distribution would balance the logits?")
# If logits[0] = logits[1], then:
# (W[0] - W[1]) @ pooled = bias[1] - bias[0]
# diff @ pooled = 0.03898
target_diff = classifier_bias[1] - classifier_bias[0]
print(f"  Need (W_neg - W_pos) @ pooled = {-target_diff:.6f}")
print(f"  Current diff mean: {diff.mean():.6f}")
print(f"  Current diff sum: {diff.sum():.6f}")

# For balanced logits with mean pooled = m:
# diff.sum() * m ≈ -target_diff
required_mean = -target_diff / diff.sum()
print(f"  Required pooled mean for balanced logits: {required_mean:.4f}")

print("\n6. Key insight:")
neg_sum = classifier_weight[0].sum()
pos_sum = classifier_weight[1].sum()
print(f"  Sum of NEG weights: {neg_sum:.4f}")
print(f"  Sum of POS weights: {pos_sum:.4f}")
print(f"  Difference: {pos_sum - neg_sum:.4f}")

if pos_sum > neg_sum:
    print(f"\n  >>> POS weights have larger sum, so positive pooled values favor POS class")
elif neg_sum > pos_sum:
    print(f"\n  >>> NEG weights have larger sum, so positive pooled values favor NEG class")
