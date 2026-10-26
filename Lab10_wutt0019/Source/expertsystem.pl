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
good_weather(sunny).
max_temp(20).
min_sleep(8).
dangerous_weather(storm).

% Condition checks
good_sleep(Sleep) :- min_sleep(Min), Sleep >= Min.
good_temp(Temp) :- max_temp(Max), Temp < Max.
tired(Sleep) :- \+ good_sleep(Sleep).
safe_weather(Weather) :- \+ dangerous_weather(Weather).

% Decide to run or not with reasoning
go_for_run(Ground, Temp, Sleep, Weather) :-
    if_then_else(tired(Sleep),
        writeln('Have not slept enough, do not run'),
        if_then_else(safe_weather(Weather),
            if_then_else(good_temp(Temp),
                if_then_else(good_ground(Ground),
                    writeln('Go for a run'),
                    writeln('Ground is not dry, do not run')),
                if_then_else(good_weather(Weather),
                    writeln('Go for a run, but it is a bit hot'),
                    writeln('It is too hot and the weather is not optimal, do not run'))),
            writeln('Weather is dangerous, do not run'))).

% Challenge Question - Allows procedural style if-then-else statements
if_then_else(Cond, Then, Else) :- (Cond -> Then; Else).