# src/process_feedback.py
import csv
import os

def summarize_feedback(path='feedback/feedback.csv'):
    if not os.path.exists(path):
        print("No feedback yet.")
        return
    counts = {}
    total = 0
    wrong = 0
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            total += 1
            _, _, fb, _ = row
            if fb.lower() in ('no', 'n'):
                wrong += 1
            counts[fb] = counts.get(fb, 0) + 1
    print("Total feedback:", total)
    print("Wrong predictions reported:", wrong)
    print("Breakdown:", counts)

if __name__ == '__main__':
    summarize_feedback()
