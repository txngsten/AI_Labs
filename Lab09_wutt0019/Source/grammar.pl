% grammar.pl - Knowledge base for simple English sentences using CFG

% Sentence structure
sentence --> noun_phrase, verb_phrase.

% Noun phrase structure
noun_phrase --> determiner, noun.
noun_phrase --> determiner, adjective, noun.

% Verb phrase structure
verb_phrase --> verb, noun_phrase.
verb_phrase --> verb.

% Lexicon
determiner --> [the].
determiner --> [a].
noun --> [cat].
noun --> [dog].
noun --> [mouse].
adjective --> [big].
adjective --> [small].
verb --> [chases].
verb --> [sees].
verb --> [likes].