"""Fishing bandit submission. Standard-library only; evaluator protocol compatible."""

import json
import math
import sys


class Strategy:
    def __init__(self, n_spots, n_casts):
        self.n_spots = n_spots
        self.n_casts = n_casts
        self.count = [0] * n_spots
        self.mean = [0.0] * n_spots
        self.m2 = [0.0] * n_spots
        self.steps = 0

    def choose(self):
        # Collect several samples from every spot before comparing estimates.
        initial_samples = min(8, max(1, self.n_casts // (4 * self.n_spots)))
        for spot in range(self.n_spots):
            if self.count[spot] < initial_samples:
                return spot

        # Near the end, prioritize the highest estimated catch.
        # Earlier, account for uncertainty as well as estimated reward.
        remaining = self.n_casts - self.steps
        exploration_weight = min(1.0, remaining / max(1, self.n_casts * 0.2))
        log_term = math.log(self.steps + 1)

        best_spot = 0
        best_score = float('-inf')
        for spot in range(self.n_spots):
            n = self.count[spot]
            variance = self.m2[spot] / (n - 1) if n > 1 else 0.0
            # Variance-aware confidence bonus; catches are measured in kg.
            uncertainty = math.sqrt(
                2.0 * max(variance, 1.0) * log_term / n
            ) + 1.5 * log_term / n
            score = self.mean[spot] + exploration_weight * uncertainty
            if score > best_score:
                best_score = score
                best_spot = spot
        return best_spot

    def update(self, spot, catch):
        self.steps += 1
        self.count[spot] += 1
        n = self.count[spot]
        delta = catch - self.mean[spot]
        self.mean[spot] += delta / n
        self.m2[spot] += delta * (catch - self.mean[spot])


def main():
    with open('test-input.json', encoding='utf-8') as file:
        rules = json.load(file)

    for _ in range(rules['n_days']):
        agent = Strategy(rules['n_spots'], rules['n_casts'])
        for _ in range(rules['n_casts']):
            spot = agent.choose()
            print(spot, flush=True)
            response = sys.stdin.readline()
            if not response:
                return
            agent.update(spot, float(response))


if __name__ == '__main__':
    main()
