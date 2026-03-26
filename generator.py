import json
import random

titles_prefix = [
    "Shadow", "Legend", "Rise", "Fall", "Return", "Last", "First",
    "Secret", "Lost", "Dark", "Bright", "Silent", "Hidden"
]

titles_suffix = [
    "Empire", "Warrior", "World", "Future", "Night", "Dream",
    "Journey", "Legacy", "Battle", "Chronicles", "Mission"
]

genres = [
    "Action", "Comedy", "Drama", "Sci-Fi", "Horror",
    "Romance", "Thriller", "Adventure", "Fantasy", "Mystery"
]

descriptions = [
    "A hero embarks on a dangerous mission to save the world.",
    "A love story unfolds in unexpected circumstances.",
    "A detective investigates a mysterious case.",
    "A group of friends face a terrifying threat.",
    "A journey through space and time reveals hidden truths.",
    "A warrior fights against powerful enemies.",
    "A young protagonist discovers their destiny.",
    "A team must survive against impossible odds.",
    "A story of betrayal, loyalty, and redemption.",
    "An adventure filled with danger and excitement."
]

def generate_movie(id):
    title = random.choice(titles_prefix) + " of the " + random.choice(titles_suffix)
    genre = random.sample(genres, k=random.randint(1, 3))
    description = random.choice(descriptions)
    year = random.randint(1980, 2024)
    rating = round(random.uniform(5.0, 9.5), 1)

    return {
        "id": id,
        "title": title,
        "genre": genre,
        "description": description,
        "year": year,
        "rating": rating
    }

movies = [generate_movie(i) for i in range(1, 1001)]

with open("movies_dataset.json", "w") as f:
    json.dump(movies, f, indent=2)

print("Dataset generated: movies_dataset.json")