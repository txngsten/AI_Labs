/**
    Student Name: Oliver Wuttke
    Student FAN: WUTT0019
    File: traffic.pl
    Date: 26-10-2026
    Description: Knowledge base for traffic conditions lab 10.
*/

% Facts about traffic patterns
typical_traffic(morning, low).
typical_traffic(afternoon, moderate).
typical_traffic(evening, high).
typical_traffic(night, low).

%Rule to check if traffic is typical
is_typical_traffic(Time, Condition) :- typical_traffic(Time, Condition).

%Rule to check if the traffic is abnormal at a given time
is_abnormal_traffic(Time, Condition) :- \+
    typical_traffic(Time, Condition).