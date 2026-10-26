/**
    Student Name: Oliver Wuttke
    Student FAN: WUTT0019
    File: expertsystem.pl
    Date: 26-10-2026
    Description: Expert system for deciding to run or not with reasoning.
*/

% Facts about running conditions
good_ground(dry).
good_weather(cloudy).
max_temp(20).        % temperature must be below this
min_sleep(8).        % hours of sleep must be at least this

% Condition checks
good_sleep(Sleep) :- min_sleep(Min), Sleep >= Min.
good_temp(Temp)   :- max_temp(Max), Temp < Max.halt

% Decide to run or not with reasoning
go_for_run(Ground, Temp, Sleep, Weather) :-
    if_then_else(good_sleep(Sleep),
        if_then_else(good_temp(Temp),
            if_then_else(good_ground(Ground),
                writeln('Go for a run'),
                writeln('Ground isn''t dry, don''t run')),
            if_then_else(good_weather(Weather),
                writeln('Go for a run, but it is a bit hot'),
                writeln('It''s too hot and the weather isn''t optimal, don''t run'))),
        writeln('Haven''t slept enough, don''t run')).

% Question 3 - Allows procedural style if-then-else statements
if_then_else(Cond, Then, Else) :- (Cond -> Then; Else).