1000 Epochs, Batch Size of 16, Hyperparameters similar to the ones suggested for tinystories, but with double the layers:
### Code Snippet #1 Prompt: "Ben was sad because":
```python
generatedStem = "Ben was sad because"
for x in tqdm(range(100)):
    generatedStem = generateOneToken(transformer, tok, generatedStem, device='cuda:0')
print(generatedStem)"
```
Transformer Output:

Ben was sad because he lost his toy.
"Lily, Lily, look at my toy!" Ben cried. "I did not know. Maybe I can fix it."
Lily looked at the toy and asked her mom for help. She was very sorry and scared. She hugged Ben and said, "It's okay, Ben. We can play together again."


Tim and Lily liked to play together. They had a big ball to play with. They threw the ball to each other. But their


### Code Snippet #2 Prompt: "Ben was happy because":

```python
generatedStem = "Ben was happy because"
for x in tqdm(range(100)):
    generatedStem = generateOneToken(transformer, tok, generatedStem, device='cuda:0')
print(generatedStem)
```
Transformer Output:

Ben was happy because he found the ball. He was so glad and happy. He thanked Lily for being so helpful, and she had a big smile on her face. She played and skipped every day and had lots of fun.


Lily liked to play with her new toy car. She had a pink car with long ago and a lot of toys. She wanted to keep her car in the box, so she started to go faster. She took a deep breath and went to her room. She felt happy


# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

