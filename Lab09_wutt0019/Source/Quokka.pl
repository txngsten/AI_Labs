/**
    Student Name: Oliver Wuttke
    Student FAN: WUTT0019
    File: Quokka.pl
    Date: 15-10-2026
    Description: Knowledge base for Quokka bus network.
*/

% Facts for travel times between cities
travel_time(adelaide, darwin, 44).
travel_time(adelaide, canberra, 22).
travel_time(adelaide, perth, 28).
travel_time(canberra, brisbane, 30).
travel_time(canberra, sydney, 5).
travel_time(canberra, melbourne, 9).

% Enables bi-directional travel
travel(X, Y, Time) :- travel_time(X, Y, Time).
travel(X, Y, Time) :- travel_time(Y, X, Time).

% Direct travel between two cities
path(X, Y, [X, Y], Time) :- travel(X, Y, Time).

% Travel between two cities with a stop
path(X, Y, Path, Time) :- path(X, Y, [X], Path, 0, Time).

% Helper predicate with visited list and time sum
    path(X, Y, Visited, [X, Y], AccTime, Time) :-
    travel(X, Y, T),
    \+ member(Y, Visited),
    Time is AccTime + T.

path(X, Y, Visited, [X | Path], AccTime, Time) :-
    travel(X, Z, T1),
    \+ member(Z, Visited),
    NewAccTime is AccTime + T1,
    path(Z, Y, [Z | Visited], Path, NewAccTime, Time).