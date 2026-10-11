/**
    Student Name: Oliver Wuttke
    Student FAN: WUTT0019
    File: panthera.pl
    Date: 10-10-2026
    Description: Knowledge base for the Genus Panthera.
*/

% Parent child relations, according to the cladogram
parent(ancestor1, paleosinesis).
parent(ancestor1, ancestor2).

parent(ancestor2, blytheae).
parent(ancestor2, snow_leopard).
parent(ancestor2, zdanskyi).
parent(ancestor2, tiger).
parent(ancestor2, european_jaguar).
parent(ancestor2, jaguar).
parent(ancestor2, ancestor3).

parent(ancestor3, leopard).
parent(ancestor3, ancestor4).

parent(ancestor4, lion).
parent(ancestor4, ancestor5).

parent(ancestor5, cave_lion).
parent(ancestor5, american_lion).
