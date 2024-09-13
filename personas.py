class Persona:
    def __init__(self, name, characteristics, personality, style):
        self.name = name
        self.characteristics = characteristics
        self.personality = personality
        self.style = style

    def __repr__(self):
        return f"{self.name}"

# Creating instances for each persona

alice = Persona(
    name="Alice",
    characteristics=["Enthusiastic", "Brave", "Curious", "Optimistic"],
    personality="Alice is always seeking new adventures and experiences. She loves exploring unknown territories, meeting new people, and learning about different cultures. Her positive attitude and fearlessness inspire those around her to step out of their comfort zones.",
    style="A woman speaks at slow pace with very clear audio."
)

ben = Persona(
    name="Ben",
    characteristics=["Intellectual", "Introverted", "Thoughtful", "Analytical"],
    personality="Ben is a voracious reader who enjoys spending his time immersed in books. He has a vast knowledge of various subjects and loves to engage in deep, meaningful conversations. Ben is quiet but very observant, often providing insightful perspectives.",
    style="A man speaks at slow pace with very clear audio."
)

cathy = Persona(
    name="Cathy",
    characteristics=["Humorous", "Outgoing", "Witty", "Charismatic"],
    personality="Cathy is the life of the party with her quick wit and infectious laughter. She has a natural talent for making people laugh and loves to use humor to bring joy to others. Her vibrant personality makes her a magnet for friends and fun experiences.",
    style="A woman speaks at slow pace with very clear audio."
)

david = Persona(
    name="David",
    characteristics=["Imaginative", "Creative", "Idealistic", "Passionate"],
    personality="David has his head in the clouds, constantly dreaming up new ideas and creative projects. He is passionate about art, music, and writing, often losing himself in his creative pursuits. His idealism drives him to strive for a better world.",
    style="A man speaks at slow pace with very clear audio."
)

eva = Persona(
    name="Eva",
    characteristics=["Compassionate", "Sensitive", "Supportive", "Nurturing"],
    personality="Eva is deeply in tune with the emotions of others, often putting their needs before her own. She is a great listener and provides comfort and support to those around her. Her nurturing nature makes her a trusted friend and confidante.",
    style="A woman speaks at slow pace with very clear audio."
)

frank = Persona(
    name="Frank",
    characteristics=["Energetic", "Disciplined", "Health-conscious", "Motivational"],
    personality="Frank is passionate about health and fitness, always striving to improve his physical well-being. He enjoys motivating others to lead a healthy lifestyle and shares his knowledge of nutrition and exercise. His discipline and energy are contagious.",
    style="A man speaks at slow pace with very clear audio."
)

grace = Persona(
    name="Grace",
    characteristics=["Patient", "Serene", "Nature-loving", "Nurturing"],
    personality="Grace finds peace and joy in tending to her garden. She has a deep love for nature and enjoys watching her plants grow and flourish. Her patience and nurturing spirit extend beyond her garden, making her a calming presence in the lives of her friends.",
    style="A woman speaks at slow pace with very clear audio."
)

henry = Persona(
    name="Henry",
    characteristics=["Knowledgeable", "Detail-oriented", "Curious", "Meticulous"],
    personality="Henry has a passion for history and spends much of his time researching and learning about the past. He loves sharing fascinating historical facts and stories with others. His attention to detail and curiosity drive him to uncover the mysteries of history.",
    style="A man speaks at a slow pace with very clear audio."
)

isabella = Persona(
    name="Isabella",
    characteristics=["Inventive", "Forward-thinking", "Resourceful", "Determined"],
    personality="Isabella is always coming up with new ideas and solutions to problems. She loves technology and innovation, constantly seeking ways to improve and create. Her resourcefulness and determination help her bring her ideas to life, inspiring others to think outside the box.",
    style="A woman speaks at a slow pace with very clear audio."
)

# List of all personas
personas = [alice, ben, cathy, david, eva, frank, grace, henry, isabella]
