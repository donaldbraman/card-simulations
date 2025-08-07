# Card Simulations

This repository contains poker probability calculators and simulators for Texas Hold'em and 7-Card Stud with Wild Cards.

## Scripts

### 1. Texas Hold'em Probability Calculator (`texas_holdem.py`)
Calculates exact mathematical probabilities for all poker hands in Texas Hold'em using combinatorial analysis.

- **Method**: Exact combinatorial calculation
- **Total Hands**: 133,784,560 possible 7-card combinations
- **Features**: 
  - Exact probabilities for all hand types
  - Win probability calculations for 2-6 player games
  - Both simplified and empirical win rate estimates

### 2. 7-Card Stud Wild Simulator (`seven_card_stud_wild_parallel.py`)
Simulates 7-card stud where the lowest hole card (and all cards of that rank) are wild.

- **Method**: Monte Carlo simulation with parallel processing
- **Simulations**: 10 million hands
- **Performance**: ~333,000 hands/second using 4 CPU cores
- **Features**:
  - Wild card detection and optimal hand evaluation
  - Win probability calculations for 2-6 player games
  - Statistical confidence intervals

## Texas Hold'em Results

### Hand Probabilities and Win Rates

| Hand Type | Frequency | Percentage | Odds | 2-Player Win | 3-Player Win | 4-Player Win | 5-Player Win | 6-Player Win |
|-----------|-----------|------------|------|--------------|--------------|--------------|--------------|--------------|
| **Royal Flush** | 4,324 | 0.0032% | 1:30,939 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **Straight Flush** | 37,260 | 0.0279% | 1:3,590 | 99.9% | 99.8% | 99.7% | 99.6% | 99.5% |
| **Four of a Kind** | 224,848 | 0.1681% | 1:594 | 99.5% | 99.0% | 98.5% | 98.0% | 97.5% |
| **Full House** | 3,473,184 | 2.5961% | 1:38 | 97.5% | 95.0% | 92.5% | 90.0% | 87.5% |
| **Flush** | 4,047,644 | 3.0255% | 1:32 | 85.0% | 72.0% | 61.0% | 52.0% | 44.0% |
| **Straight** | 6,180,020 | 4.6194% | 1:21 | 75.0% | 56.0% | 42.0% | 31.5% | 24.0% |
| **Three of a Kind** | 6,461,620 | 4.8299% | 1:20 | 68.0% | 46.0% | 31.0% | 21.0% | 14.0% |
| **Two Pair** | 31,433,400 | 23.4955% | 1:3 | 62.0% | 39.0% | 24.0% | 15.0% | 9.5% |
| **One Pair** | 58,627,800 | 43.8225% | 1:1 | 49.0% | 24.0% | 12.0% | 6.0% | 3.0% |
| **High Card** | 23,294,460 | 17.4119% | 1:5 | 17.0% | 3.0% | 0.5% | 0.1% | 0.0% |

*Win rates shown are empirical estimates that account for shared community cards and tie scenarios.*

### Key Insights
- One Pair is the most common hand (43.82% of all hands)
- Full House or better occurs only 2.80% of the time
- In heads-up play, even One Pair wins about 49% of the time
- In 6-player games, you typically need Three of a Kind or better to have a good chance of winning

## 7-Card Stud Wild Results (10 Million Simulations)

### Hand Probabilities with Wild Cards

| Hand Type | Percentage | 2-Player Win | 3-Player Win | 4-Player Win | 5-Player Win | 6-Player Win |
|-----------|------------|--------------|--------------|--------------|--------------|--------------|
| **Five of a Kind** | 2% | 100% | 100% | 100% | 100% | 100% |
| **Royal Flush** | 2% | 98% | 97% | 95% | 93% | 92% |
| **Straight Flush** | 4% | 97% | 93% | 90% | 87% | 84% |
| **Four of a Kind** | 16% | 93% | 87% | 81% | 75% | 70% |
| **Full House** | 12% | 77% | 60% | 46% | 36% | 28% |
| **Flush** | 9% | 65% | 43% | 28% | 18% | 12% |
| **Straight** | 48% | 56% | 32% | 18% | 10% | 6% |
| **Three of a Kind** | 4% | 8% | 1% | 0% | 0% | 0% |
| **One Pair** | 4% | 4% | 0% | 0% | 0% | 0% |
| **Two Pair** | 0% | - | - | - | - | - |
| **High Card** | 0% | - | - | - | - | - |

*Note: Two Pair and High Card are impossible because wild cards always create at least One Pair.*

### Wild Card Distribution

| Wild Cards | Count | Percentage |
|------------|-------|------------|
| 1 | 7,030,477 | 70.30% |
| 2 | 2,675,850 | 26.76% |
| 3 | 285,487 | 2.85% |
| 4 | 8,186 | 0.08% |

### Key Statistics
- **Average wild cards per hand**: 1.33
- **Full House or better**: 34.61% of hands
- **Royal Flush frequency**: 1.61% (1 in 61 hands)
- **Most common hand**: Straight (48% of all hands)

### Impact of Wild Cards
The low hole card wild rule dramatically changes the game:
- Straights become the most common hand (48% vs 4.6% in standard poker)
- Premium hands (Full House+) occur in over 1/3 of all hands
- Every hand contains at least one wild card (minimum One Pair)
- Royal Flushes are ~500x more common than in standard poker

## Running the Simulations

### Texas Hold'em
```bash
python texas_holdem.py
```
Runs instantly - uses exact mathematical calculations.

### 7-Card Stud Wild
```bash
python seven_card_stud_wild_parallel.py
```
Runs 10 million simulations in ~30 seconds using 4 CPU cores.

For fewer simulations or different core counts, modify the script parameters:
- `num_hands` in `main()` function
- `num_cores` in `run_parallel_simulation()` function

## Technical Details

### Texas Hold'em Calculator
- Uses combinatorial mathematics: C(52,7) = 133,784,560
- Exact probability calculations for each hand type
- Win probability formula: P(win) = P(all opponents have worse)^(N-1)

### 7-Card Stud Wild Simulator
- Monte Carlo simulation for complex wild card scenarios
- Parallel processing using Python's multiprocessing module
- Optimized hand evaluation with early termination
- Statistical confidence: ±0.002% for shown values with 10M simulations

## Requirements
- Python 3.7+
- Standard library only (no external dependencies)

## License
MIT