"""
7-Card Stud with Low Hole Card Wild - Fixed Royal Flush Detection

This version properly detects Royal Flushes by checking if wild cards
can complete 10-J-Q-K-A in any suit.
"""

import random
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set, Optional
from enum import IntEnum
import time
from itertools import combinations, product


class Rank(IntEnum):
    """Card ranks with 2 being lowest, Ace being highest"""
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


class Suit(IntEnum):
    """Card suits"""
    CLUBS = 1
    DIAMONDS = 2
    HEARTS = 3
    SPADES = 4


class Card:
    """Represents a playing card"""
    
    def __init__(self, rank: Rank, suit: Suit):
        self.rank = rank
        self.suit = suit
    
    def __repr__(self):
        rank_str = {
            Rank.TWO: '2', Rank.THREE: '3', Rank.FOUR: '4', Rank.FIVE: '5',
            Rank.SIX: '6', Rank.SEVEN: '7', Rank.EIGHT: '8', Rank.NINE: '9',
            Rank.TEN: 'T', Rank.JACK: 'J', Rank.QUEEN: 'Q', 
            Rank.KING: 'K', Rank.ACE: 'A'
        }[self.rank]
        suit_str = {
            Suit.CLUBS: '♣', Suit.DIAMONDS: '♦', 
            Suit.HEARTS: '♥', Suit.SPADES: '♠'
        }[self.suit]
        return f"{rank_str}{suit_str}"
    
    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self):
        return hash((self.rank, self.suit))


class HandType(IntEnum):
    """Hand rankings from lowest to highest"""
    HIGH_CARD = 1
    ONE_PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10
    FIVE_OF_A_KIND = 11  # Only possible with wild cards


class FixedWildCardEvaluator:
    """Evaluator with proper Royal Flush detection"""
    
    @staticmethod
    def identify_wild_rank(hole_cards: List[Card]) -> Rank:
        """Returns the rank of the lowest hole card"""
        return min(card.rank for card in hole_cards)
    
    @staticmethod
    def count_wilds(hand: List[Card], wild_rank: Rank) -> int:
        """Counts how many wild cards are in the hand"""
        return sum(1 for card in hand if card.rank == wild_rank)
    
    @staticmethod
    def can_make_royal_flush(non_wild_cards: List[Card], num_wilds: int) -> bool:
        """Check if we can make a royal flush with the given cards and wilds"""
        royal_ranks = {Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE}
        
        # Check each suit
        for suit in Suit:
            # Count royal cards we have in this suit
            suited_royals = [c for c in non_wild_cards 
                           if c.suit == suit and c.rank in royal_ranks]
            
            # Can we complete the royal flush with wilds?
            if len(suited_royals) + num_wilds >= 5:
                # Need to verify we don't need more than 5 cards total
                needed = 5 - len(suited_royals)
                if needed <= num_wilds:
                    return True
        
        return False
    
    @staticmethod
    def is_straight(ranks: List[int]) -> Tuple[bool, int]:
        """Check if 5 ranks form a straight. Returns (is_straight, high_card)"""
        if len(ranks) != 5:
            return False, 0
        
        unique_ranks = sorted(set(ranks))
        if len(unique_ranks) != 5:
            return False, 0
        
        # Check regular straight
        if unique_ranks[-1] - unique_ranks[0] == 4:
            return True, unique_ranks[-1]
        
        # Check A-2-3-4-5 (wheel)
        if unique_ranks == [2, 3, 4, 5, 14]:
            return True, 5
        
        return False, 0
    
    @staticmethod
    def evaluate_5_card_hand(cards: List[Card]) -> HandType:
        """Evaluates exactly 5 cards and returns the hand type."""
        if len(cards) != 5:
            raise ValueError("Must evaluate exactly 5 cards")
        
        ranks = [card.rank for card in cards]
        suits = [card.suit for card in cards]
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)
        
        is_flush = len(suit_counts) == 1
        is_str, str_high = FixedWildCardEvaluator.is_straight(ranks)
        
        # Count patterns
        counts = sorted(rank_counts.values(), reverse=True)
        
        # Determine hand type
        if counts == [5]:
            return HandType.FIVE_OF_A_KIND
        elif is_str and is_flush:
            # Check specifically for Royal Flush: 10-J-Q-K-A of same suit
            if sorted(ranks) == [10, 11, 12, 13, 14]:
                return HandType.ROYAL_FLUSH
            else:
                return HandType.STRAIGHT_FLUSH
        elif counts == [4, 1]:
            return HandType.FOUR_OF_A_KIND
        elif counts == [3, 2]:
            return HandType.FULL_HOUSE
        elif is_flush:
            return HandType.FLUSH
        elif is_str:
            return HandType.STRAIGHT
        elif counts == [3, 1, 1]:
            return HandType.THREE_OF_A_KIND
        elif counts == [2, 2, 1]:
            return HandType.TWO_PAIR
        elif counts == [2, 1, 1, 1]:
            return HandType.ONE_PAIR
        else:
            return HandType.HIGH_CARD
    
    @staticmethod
    def evaluate_with_wilds(hand: List[Card], wild_rank: Rank) -> HandType:
        """Evaluates the best possible hand with wild cards."""
        # Identify wild and non-wild cards
        wild_cards = [c for c in hand if c.rank == wild_rank]
        non_wild_cards = [c for c in hand if c.rank != wild_rank]
        num_wilds = len(wild_cards)
        
        if num_wilds == 0:
            # No wilds, evaluate all 5-card combinations
            best_hand = HandType.HIGH_CARD
            for combo in combinations(hand, 5):
                hand_type = FixedWildCardEvaluator.evaluate_5_card_hand(list(combo))
                best_hand = max(best_hand, hand_type)
            return best_hand
        
        # Quick check for guaranteed hands
        if num_wilds >= 4:
            return HandType.FIVE_OF_A_KIND
        
        # CHECK FOR ROYAL FLUSH FIRST
        if FixedWildCardEvaluator.can_make_royal_flush(non_wild_cards, num_wilds):
            # Verify by trying to make the royal flush
            for suit in Suit:
                royal_ranks = {Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE}
                suited_royals = [c for c in non_wild_cards 
                               if c.suit == suit and c.rank in royal_ranks]
                
                if len(suited_royals) + num_wilds >= 5:
                    needed_ranks = royal_ranks - {c.rank for c in suited_royals}
                    if len(needed_ranks) <= num_wilds:
                        # We can make a royal flush!
                        return HandType.ROYAL_FLUSH
        
        # Continue with other hand evaluations
        best_hand = HandType.HIGH_CARD
        
        if num_wilds == 3:
            # With 3 wilds, check for Five of a Kind first
            rank_counts = Counter(c.rank for c in non_wild_cards)
            if any(count >= 2 for count in rank_counts.values()):
                return HandType.FIVE_OF_A_KIND
            
            # Check for straight flush
            suits_in_hand = defaultdict(list)
            for c in non_wild_cards:
                suits_in_hand[c.suit].append(c.rank)
            
            for suit, ranks in suits_in_hand.items():
                if len(ranks) >= 2:
                    # Can likely make a straight flush
                    return HandType.STRAIGHT_FLUSH
            
            # Otherwise at least Four of a Kind
            return HandType.FOUR_OF_A_KIND
        
        elif num_wilds == 2:
            # With 2 wilds, check patterns
            rank_counts = Counter(c.rank for c in non_wild_cards)
            suit_counts = Counter(c.suit for c in non_wild_cards)
            
            max_rank_count = max(rank_counts.values()) if rank_counts else 0
            max_suit_count = max(suit_counts.values()) if suit_counts else 0
            
            if max_rank_count >= 3:
                return HandType.FIVE_OF_A_KIND
            elif max_rank_count == 2:
                pairs = [r for r, c in rank_counts.items() if c == 2]
                if len(pairs) >= 2:
                    return HandType.FULL_HOUSE
                return HandType.FOUR_OF_A_KIND
            elif max_suit_count >= 3:
                # Good chance for flush, check for straight flush
                best_hand = HandType.FLUSH
                for suit in Suit:
                    suited_cards = [c for c in non_wild_cards if c.suit == suit]
                    if len(suited_cards) >= 3:
                        suited_ranks = sorted([c.rank for c in suited_cards])
                        # With 2 wilds, check if we can make straight flush
                        if len(suited_ranks) >= 3 and suited_ranks[-1] - suited_ranks[0] <= 6:
                            return HandType.STRAIGHT_FLUSH
            
            # Check for straight
            all_ranks = sorted([c.rank for c in non_wild_cards])
            if len(all_ranks) >= 3:
                # With 2 wilds, we can likely make a straight
                for i in range(len(all_ranks) - 2):
                    if all_ranks[i+2] - all_ranks[i] <= 6:
                        best_hand = max(best_hand, HandType.STRAIGHT)
            
            # At minimum with 2 wilds
            best_hand = max(best_hand, HandType.THREE_OF_A_KIND)
            
        elif num_wilds == 1:
            # With 1 wild, check patterns
            rank_counts = Counter(c.rank for c in non_wild_cards)
            suit_counts = Counter(c.suit for c in non_wild_cards)
            
            max_rank_count = max(rank_counts.values()) if rank_counts else 0
            max_suit_count = max(suit_counts.values()) if suit_counts else 0
            
            if max_rank_count >= 4:
                return HandType.FIVE_OF_A_KIND
            elif max_rank_count == 3:
                return HandType.FOUR_OF_A_KIND
            elif max_rank_count == 2:
                pairs = sum(1 for c in rank_counts.values() if c == 2)
                if pairs >= 2:
                    return HandType.FULL_HOUSE
                else:
                    best_hand = HandType.THREE_OF_A_KIND
            
            # Check for flush
            if max_suit_count >= 4:
                best_hand = max(best_hand, HandType.FLUSH)
                # Check for straight flush
                suit = [s for s, c in suit_counts.items() if c >= 4][0]
                suited_ranks = sorted([c.rank for c in non_wild_cards if c.suit == suit])
                if len(suited_ranks) >= 4:
                    for i in range(len(suited_ranks) - 3):
                        if suited_ranks[i+3] - suited_ranks[i] <= 4:
                            return HandType.STRAIGHT_FLUSH
            
            # Check for straight
            all_ranks = sorted([c.rank for c in non_wild_cards])
            for i in range(max(0, len(all_ranks) - 3)):
                if i + 3 < len(all_ranks) and all_ranks[i+3] - all_ranks[i] <= 5:
                    best_hand = max(best_hand, HandType.STRAIGHT)
            
            # At minimum with 1 wild
            best_hand = max(best_hand, HandType.ONE_PAIR)
        
        return best_hand


class FixedSevenCardStudWildSimulator:
    """Simulator with fixed Royal Flush detection"""
    
    def __init__(self):
        self.deck = self._create_deck()
        self.hand_counts = defaultdict(int)
        self.wild_count_distribution = defaultdict(int)
        
    def _create_deck(self) -> List[Card]:
        """Creates a standard 52-card deck"""
        deck = []
        for rank in Rank:
            for suit in Suit:
                deck.append(Card(rank, suit))
        return deck
    
    def deal_hand(self) -> Tuple[List[Card], List[Card]]:
        """Deals a 7-card stud hand (3 hole + 4 exposed)"""
        shuffled = self.deck.copy()
        random.shuffle(shuffled)
        hole_cards = shuffled[:3]
        exposed_cards = shuffled[3:7]
        return hole_cards, exposed_cards
    
    def simulate_hand(self) -> Tuple[HandType, int]:
        """Simulates one hand and returns the best hand type and wild count."""
        hole_cards, exposed_cards = self.deal_hand()
        full_hand = hole_cards + exposed_cards
        
        # Identify wild rank and count
        wild_rank = FixedWildCardEvaluator.identify_wild_rank(hole_cards)
        wild_count = FixedWildCardEvaluator.count_wilds(full_hand, wild_rank)
        
        # Evaluate best possible hand
        best_hand = FixedWildCardEvaluator.evaluate_with_wilds(full_hand, wild_rank)
        
        return best_hand, wild_count
    
    def run_simulation(self, num_hands: int = 1000000) -> Dict:
        """Runs Monte Carlo simulation for specified number of hands."""
        print(f"Running simulation with FIXED Royal Flush detection...")
        print(f"Simulating {num_hands:,} hands...")
        print()
        
        start_time = time.time()
        
        for i in range(num_hands):
            if i % 50000 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = (num_hands - i) / rate
                print(f"  Processed {i:,} hands... (~{remaining:.0f} seconds remaining)")
                # Show interim Royal Flush count
                if HandType.ROYAL_FLUSH in self.hand_counts:
                    print(f"    Royal Flushes so far: {self.hand_counts[HandType.ROYAL_FLUSH]}")
            
            hand_type, wild_count = self.simulate_hand()
            self.hand_counts[hand_type] += 1
            self.wild_count_distribution[wild_count] += 1
        
        elapsed = time.time() - start_time
        print(f"\nSimulation complete in {elapsed:.1f} seconds")
        print(f"Rate: {num_hands/elapsed:.0f} hands/second\n")
        
        # Calculate probabilities
        results = {}
        for hand_type in HandType:
            count = self.hand_counts[hand_type]
            probability = count / num_hands
            results[hand_type] = {
                'count': count,
                'probability': probability,
                'percentage': probability * 100
            }
        
        # Wild count statistics
        wild_stats = {}
        for wild_count in sorted(self.wild_count_distribution.keys()):
            count = self.wild_count_distribution[wild_count]
            probability = count / num_hands
            wild_stats[wild_count] = {
                'count': count,
                'probability': probability,
                'percentage': probability * 100
            }
        
        return {'hands': results, 'wilds': wild_stats}


def format_results(results: Dict, num_hands: int):
    """Formats and displays the simulation results"""
    
    print("=" * 75)
    print("7-CARD STUD WITH LOW HOLE CARD WILD - FINAL RESULTS")
    print("=" * 75)
    print()
    
    print(f"Total hands simulated: {num_hands:,}")
    print()
    
    # Hand type distribution
    print("=" * 75)
    print("HAND TYPE PROBABILITIES (Ranked from Highest to Lowest)")
    print("-" * 75)
    print(f"{'Rank':<5} {'Hand Type':<20} {'Frequency':>10} {'Probability':>12} {'Percentage':>10} {'Odds':>12}")
    print("-" * 75)
    
    hand_stats = results['hands']
    hand_names = {
        HandType.HIGH_CARD: "High Card",
        HandType.ONE_PAIR: "One Pair",
        HandType.TWO_PAIR: "Two Pair",
        HandType.THREE_OF_A_KIND: "Three of a Kind",
        HandType.STRAIGHT: "Straight",
        HandType.FLUSH: "Flush",
        HandType.FULL_HOUSE: "Full House",
        HandType.FOUR_OF_A_KIND: "Four of a Kind",
        HandType.STRAIGHT_FLUSH: "Straight Flush",
        HandType.ROYAL_FLUSH: "Royal Flush",
        HandType.FIVE_OF_A_KIND: "Five of a Kind"
    }
    
    # Display from best to worst with rank numbers
    rank = 1
    for hand_type in reversed(list(HandType)):
        stats = hand_stats[hand_type]
        
        # Only show hands that actually occur
        if stats['count'] > 0:
            if stats['probability'] > 0:
                odds = f"1:{int(1/stats['probability']-1):,}"
            else:
                odds = "N/A"
            
            # Highlight Royal Flush if it occurs
            if hand_type == HandType.ROYAL_FLUSH and stats['count'] > 0:
                print(f"{rank:<5} {hand_names[hand_type]:<20} {stats['count']:>10,} {stats['probability']:>12.8f} "
                      f"{stats['percentage']:>9.4f}% {odds:>12} *")
            else:
                print(f"{rank:<5} {hand_names[hand_type]:<20} {stats['count']:>10,} {stats['probability']:>12.8f} "
                      f"{stats['percentage']:>9.4f}% {odds:>12}")
            rank += 1
    
    print("-" * 75)
    if HandType.ROYAL_FLUSH in hand_stats and hand_stats[HandType.ROYAL_FLUSH]['count'] > 0:
        print("* Royal Flush now properly detected!")
    
    # Wild card distribution
    print()
    print("=" * 75)
    print("WILD CARD DISTRIBUTION")
    print("-" * 75)
    print(f"{'Wild Cards':<15} {'Count':>12} {'Percentage':>10}")
    print("-" * 75)
    
    wild_stats = results['wilds']
    for wild_count in sorted(wild_stats.keys()):
        stats = wild_stats[wild_count]
        print(f"{wild_count:^15} {stats['count']:>12,} {stats['percentage']:>9.4f}%")
    
    # Summary statistics
    print()
    print("=" * 75)
    print("SUMMARY STATISTICS")
    print("-" * 75)
    
    # Average wild cards
    total_wilds = sum(wild_count * stats['count'] for wild_count, stats in wild_stats.items())
    avg_wilds = total_wilds / num_hands
    print(f"• Average wild cards per hand: {avg_wilds:.2f}")
    
    # Premium hands
    premium_hands = sum(stats['count'] for hand_type, stats in hand_stats.items() 
                       if hand_type >= HandType.FULL_HOUSE)
    premium_pct = (premium_hands / num_hands) * 100
    print(f"• Full House or better: {premium_pct:.2f}%")
    
    # Strong hands
    strong_hands = sum(stats['count'] for hand_type, stats in hand_stats.items() 
                      if hand_type >= HandType.STRAIGHT)
    strong_pct = (strong_hands / num_hands) * 100
    print(f"• Straight or better: {strong_pct:.2f}%")
    
    # Royal Flush specific
    if HandType.ROYAL_FLUSH in hand_stats:
        rf_count = hand_stats[HandType.ROYAL_FLUSH]['count']
        if rf_count > 0:
            print(f"• Royal Flushes: {rf_count} ({hand_stats[HandType.ROYAL_FLUSH]['percentage']:.4f}%)")


def main():
    """Main function to run the fixed simulation"""
    
    # Number of hands to simulate
    num_hands = 1000000  # 1 million for good accuracy
    
    # Create simulator and run
    simulator = FixedSevenCardStudWildSimulator()
    results = simulator.run_simulation(num_hands)
    
    # Display results
    format_results(results, num_hands)


if __name__ == "__main__":
    main()