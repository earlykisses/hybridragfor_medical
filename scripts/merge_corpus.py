import json

files = [
    "data/corpus.jsonl",
    "data/textbooks_corpus.jsonl",
    "data/statpearls_corpus.jsonl"
]

output_file = "data/merged_corpus.jsonl"

count = 0

with open(
    output_file,
    "w",
    encoding="utf-8"
) as outfile:

    for file in files:

        print(f"Processing {file}")

        with open(
            file,
            "r",
            encoding="utf-8"
        ) as infile:

            for line in infile:

                outfile.write(line)

                count += 1

print("\nDone.")
print(f"Total passages: {count}")
print(f"Saved to: {output_file}")