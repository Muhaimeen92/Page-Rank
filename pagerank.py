import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    prob_dist = {}

    #Probability of choosing any page in the corpus from current page, 'page'
    m = len(corpus)
    for link in corpus:
        prob_dist[link] = (1 - damping_factor) / m

    #Probability of choosing a page from the links in current page, 'page
    n = int()
    for links in page.values():
        n = len(links)
    if n == 0: #if the current page, 'page', has no outgoing links, then return a probability distrubution with equal probability of choosing any page in the corpus
        for link in corpus:
            prob_dist[link] = 1 / m
        return prob_dist
    for links in page.values():
        for link in links:
            prob_dist[link] += (damping_factor / n)

    return prob_dist

    """# Checking to see if there are any pages with no links
    for x, y in corpus.items():
        if len(y) == 0:
            for pages in corpus:
                prob_dist[pages] = 1 / len(corpus)
            return prob_dist

    # Calculating probability based on damping factor
    for m, n in page.items():
        probability = ((1 - damping_factor) / (1 + len(n)))
        prob_dist[m] = probability
        for item in n:
            prob_dist[item] = damping_factor / len(n) + probability
    return prob_dist"""



def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # First pick is random from all the pages in the corpus
    rand_no1 = random.randrange(1, len(corpus) + 1, 1)
    i = 0
    page = {}
    for link in corpus:
        i += 1
        if i == rand_no1:
            page[link] = corpus[link]
            break


    PageRank = {}

    for j in range (n):
        prob_dist = transition_model(corpus, page, damping_factor) # Get the probability distribution for the selected page
        selection = sampling(prob_dist) # Based on the probability distribution make a random selection
        if selection in PageRank:
            PageRank[selection] += 1
        else:
            PageRank[selection] = 1
        page = {}
        page[selection] = corpus[selection]

    for k, l in PageRank.items():
        PageRank[k] = l / n

    total = 0
    for values in PageRank.values():
        total += values
    print(total)
    return PageRank


def sampling(prob_dist):

    """
    Generating a random real number between 0 and 1. Then assigning each pages relative probability
    with a range and if the random number is in that range, select that page.
    """
    rand_no2 = random.random()
    page_links = []
    page_prob = []

    for links in prob_dist:
        page_links.append(links)

    for probability in prob_dist.values():
        page_prob.append(probability)

    low_range = 0
    up_range = 0
    page_selected = str()
    for i in range(len(page_links)):
        up_range += page_prob[i]
        if rand_no2 > low_range and rand_no2 <= up_range:
            page_selected = page_links[i]
            break
        else:
            low_range = page_prob[i]

    return(page_selected)


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    PageRank = {}
    old_PR = {}
    N = len(corpus)
    incoming_set = {}

    for page in corpus:
        PageRank[page] = 1 / N
        old_PR[page] = 1 / N
        if len(corpus[page]) == 0:
            corpus[page] = set(p for p in corpus)

    for page in corpus:

        pages_to_add = set()

        for pages2, links in corpus.items():
            if pages2 == page:
                continue
            if page in links:
                pages_to_add.add(pages2)
            incoming_set[page] = pages_to_add


    while True:
        count = 0
        for page in corpus:

            sum_PRI = float()
            if len(incoming_set[page]) == 0:
                sum_PRI = 0
            else:
                for link in incoming_set[page]:
                    if len(corpus[link]) == 0:
                        sum_PRI += PageRank[link] / N
                    else:
                        sum_PRI += PageRank[link] / len(corpus[link])

            PageRank[page] = (1 - damping_factor) / N + damping_factor * (sum_PRI)

        for pages in corpus:
            if abs(old_PR[pages] - PageRank[pages]) < 0.001:
                count += 1
            old_PR[pages] = PageRank[pages]

        if count == len(corpus):
            break

    return PageRank


if __name__ == "__main__":
    main()
