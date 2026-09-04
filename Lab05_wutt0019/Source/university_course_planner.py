"""
Student Name: Oliver Wuttke
Student FAN: WUTT0019
File: university_course_planner.py
Date: 04-09-2026
Description: Simple knowledge base and reasoning for a university course planner, using real Flinders courses.
"""

# Returns true if prerequisites are met for a course
def met_requirements(KB, courses_taken, course):
    return all(p in courses_taken for p in KB.get(course, []))

# Returns a list of recommended courses based on completed ones
def suggest_course(KB, courses_taken):
    suggestions = []
    for topic, prerequisites in KB.items():
        if topic not in courses_taken:
            for pre in prerequisites:
                if pre not in courses_taken:
                    suggestions.append(pre)

    return suggestions

# Topics mapped to list of prerequisites
KB = {
    "COMP3742" : ["COMP2712", "COMP2711"],
    "COMP2712" : ["COMP1102"],
    "COMP2711" : ["COMP1102"]
}

courses_taken = {"COMP1102"}

print(suggest_course(KB, courses_taken))



