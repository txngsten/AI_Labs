/**
    Student Name: Oliver Wuttke
    Student FAN: WUTT0019
    File: grammar.pl
    Date: 15-10-2026
    Description: Knowledge base for context-free grammar for lab 9.
*/

% grammar.pl - Knowledge base for simple English sentences using CFG

% Sentence structure
sentence --> prepositional_phrase, noun_phrase, verb_phrase.

% Noun phrase structure
noun_phrase --> determiner, noun.
noun_phrase --> determiner, adjective, noun.

% Verb phrase structure
verb_phrase --> verb, prepositional_phrase, noun_phrase.
verb_phrase --> verb.

% Prepositional phrase structure
prepositional_phrase --> preposition, noun_phrase.

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
preposition --> [on].
preposition --> [with].
preposition --> [under].