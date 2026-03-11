"""
risk_advisor.py - Academic Risk Advisor Agent Module.

This module implements the rule-based Academic Risk Advisor Agent.
The agent analyzes ML predictions along with student attributes
to generate personalized academic recommendations.

How it works:
1. Takes the predicted risk level and student's data as input
2. Applies a set of predefined academic rules
3. Generates actionable recommendations for faculty
4. Prioritizes recommendations by urgency

This is a rule-based expert system (not a neural network).
Rules are based on common educational intervention strategies.
"""


def generate_recommendations(student_data, risk_level, pass_probability):
    """
    Generate personalized academic recommendations based on student data
    and predicted risk level.
    
    The agent checks multiple conditions and generates relevant advice.
    Each recommendation has a priority level and category.
    
    Args:
        student_data (dict): Dictionary of student attributes
            Keys include: absences, studytime, G1, G2, failures, 
            goout, Dalc, Walc, health, internet, higher, etc.
        risk_level (str): "High Risk", "Medium Risk", or "Low Risk"
        pass_probability (float): Probability of passing (0.0 to 1.0)
    
    Returns:
        list: List of recommendation dictionaries, each containing:
            - 'category': Type of recommendation (Attendance, Academic, etc.)
            - 'priority': Urgency level (Critical, Important, Suggested)
            - 'recommendation': The actual advice text
            - 'icon': Emoji icon for visual display
    """
    recommendations = []  # Store all generated recommendations
    
    # ================================================================
    # RULE 1: ATTENDANCE-BASED RECOMMENDATIONS
    # High absences indicate attendance issues that need addressing
    # ================================================================
    absences = student_data.get("absences", 0)
    
    if absences > 20:
        # More than 20 absences is a serious red flag
        recommendations.append({
            "category": "Attendance",
            "priority": "Critical",
            "recommendation": (
                f"Student has {absences} absences (very high). "
                "Immediate counseling recommended. Consider parent-teacher meeting. "
                "Investigate potential personal/health issues affecting attendance."
            ),
            "icon": "🚨"
        })
    elif absences > 10:
        # 10-20 absences need monitoring
        recommendations.append({
            "category": "Attendance",
            "priority": "Important",
            "recommendation": (
                f"Student has {absences} absences (above average). "
                "Monitor attendance closely. Assign attendance mentor. "
                "Send regular attendance reports to guardian."
            ),
            "icon": "⚠️"
        })
    elif absences > 5:
        # 5-10 absences - mild warning
        recommendations.append({
            "category": "Attendance",
            "priority": "Suggested",
            "recommendation": (
                f"Student has {absences} absences. "
                "Encourage consistent attendance. "
                "Discuss importance of regular classroom participation."
            ),
            "icon": "📋"
        })
    
    # ================================================================
    # RULE 2: STUDY TIME-BASED RECOMMENDATIONS
    # studytime: 1=<2hrs, 2=2-5hrs, 3=5-10hrs, 4=>10hrs
    # ================================================================
    studytime = student_data.get("studytime", 2)
    
    if studytime <= 1:
        # Very low study time - needs intervention
        recommendations.append({
            "category": "Study Habits",
            "priority": "Critical" if risk_level == "High Risk" else "Important",
            "recommendation": (
                "Student studies less than 2 hours per week. "
                "Assign a structured study plan with daily targets. "
                "Pair with a high-performing study buddy. "
                "Provide access to additional learning resources."
            ),
            "icon": "📚"
        })
    elif studytime == 2 and risk_level in ["High Risk", "Medium Risk"]:
        # Moderate study time but at risk - encourage more
        recommendations.append({
            "category": "Study Habits",
            "priority": "Suggested",
            "recommendation": (
                "Student studies 2-5 hours per week. "
                "For improvement, recommend increasing to at least 5-10 hours. "
                "Suggest time management workshops."
            ),
            "icon": "⏰"
        })
    
    # ================================================================
    # RULE 3: GRADE TREND ANALYSIS (G1 vs G2)
    # G1 = first period grade, G2 = second period grade (both 0-20)
    # Declining grades indicate worsening performance
    # ================================================================
    g1 = student_data.get("G1", 10)
    g2 = student_data.get("G2", 10)
    
    if g2 < g1 - 2:
        # Grade dropped by more than 2 points - declining trend
        recommendations.append({
            "category": "Academic Performance",
            "priority": "Critical",
            "recommendation": (
                f"⚡ Declining grade trend detected! "
                f"G1: {g1}/20 → G2: {g2}/20 (dropped by {g1 - g2} points). "
                "Immediate academic intervention needed. "
                "Consider remedial classes and extra tutoring sessions."
            ),
            "icon": "📉"
        })
    elif g2 > g1 + 2:
        # Grade improved significantly - positive reinforcement
        recommendations.append({
            "category": "Academic Performance",
            "priority": "Suggested",
            "recommendation": (
                f"✅ Positive grade improvement detected! "
                f"G1: {g1}/20 → G2: {g2}/20 (improved by {g2 - g1} points). "
                "Encourage the student to maintain this trajectory. "
                "Consider recognition or awards for improvement."
            ),
            "icon": "📈"
        })
    
    # ================================================================
    # RULE 4: LOW INTERNAL MARKS RECOMMENDATIONS
    # If current grades are below passing threshold
    # ================================================================
    if g2 < 10:
        # Below passing threshold in recent assessment
        recommendations.append({
            "category": "Remedial Action",
            "priority": "Critical",
            "recommendation": (
                f"Student's recent grade ({g2}/20) is below passing threshold. "
                "Enroll in remedial classes immediately. "
                "Assign additional practice assignments. "
                "Schedule one-on-one tutoring sessions."
            ),
            "icon": "🔴"
        })
    
    # ================================================================
    # RULE 5: PAST FAILURES ANALYSIS
    # failures: number of times student has failed before
    # ================================================================
    failures = student_data.get("failures", 0)
    
    if failures >= 2:
        # Multiple past failures - chronic issue
        recommendations.append({
            "category": "Academic History",
            "priority": "Critical",
            "recommendation": (
                f"Student has {failures} past class failures. "
                "This indicates chronic academic difficulties. "
                "Recommend comprehensive academic support program. "
                "Assign a dedicated academic advisor. "
                "Consider learning disability assessment."
            ),
            "icon": "🔄"
        })
    elif failures == 1:
        # One past failure - needs attention
        recommendations.append({
            "category": "Academic History",
            "priority": "Important",
            "recommendation": (
                "Student has 1 past failure. "
                "Closely monitor to prevent recurrence. "
                "Provide targeted support in weak subject areas."
            ),
            "icon": "⚡"
        })
    
    # ================================================================
    # RULE 6: SOCIAL BEHAVIOR ANALYSIS
    # goout: frequency of going out (1-5)
    # Dalc: workday alcohol consumption (1-5)
    # Walc: weekend alcohol consumption (1-5)
    # ================================================================
    goout = student_data.get("goout", 3)
    dalc = student_data.get("Dalc", 1)
    walc = student_data.get("Walc", 1)
    
    if dalc >= 3 or walc >= 4:
        # High alcohol consumption is a concern
        recommendations.append({
            "category": "Behavioral",
            "priority": "Critical",
            "recommendation": (
                "High alcohol consumption pattern detected. "
                "This significantly impacts academic performance. "
                "Recommend confidential counseling session. "
                "Inform guardian with sensitivity."
            ),
            "icon": "🏥"
        })
    
    if goout >= 4 and risk_level in ["High Risk", "Medium Risk"]:
        # Excessive social outings while at risk
        recommendations.append({
            "category": "Behavioral",
            "priority": "Important",
            "recommendation": (
                "Student frequently goes out with friends while at academic risk. "
                "Discuss balance between social life and academics. "
                "Help create a balanced weekly schedule."
            ),
            "icon": "🎯"
        })
    
    # ================================================================
    # RULE 7: SUPPORT SYSTEM CHECK
    # schoolsup: extra educational support
    # famsup: family educational support
    # ================================================================
    schoolsup = student_data.get("schoolsup", 0)
    famsup = student_data.get("famsup", 0)
    
    # schoolsup and famsup may be encoded (0/1) or text (yes/no)
    if risk_level == "High Risk":
        if schoolsup in [0, "no"]:
            recommendations.append({
                "category": "Support System",
                "priority": "Critical",
                "recommendation": (
                    "High-risk student is NOT receiving extra school support. "
                    "Immediately enroll in school support program. "
                    "Assign dedicated tutor for weak subjects."
                ),
                "icon": "🆘"
            })
        if famsup in [0, "no"]:
            recommendations.append({
                "category": "Support System",
                "priority": "Important",
                "recommendation": (
                    "Student lacks family educational support. "
                    "Engage family through parent orientation. "
                    "Provide resources for home-based academic help."
                ),
                "icon": "👨‍👩‍👧"
            })
    
    # ================================================================
    # RULE 8: HEALTH-BASED RECOMMENDATIONS
    # health: current health status (1-5, where 1=very bad, 5=very good)
    # ================================================================
    health = student_data.get("health", 3)
    
    if health <= 2:
        recommendations.append({
            "category": "Health & Wellbeing",
            "priority": "Important",
            "recommendation": (
                "Student reports poor health status. "
                "Health issues can significantly impact academic performance. "
                "Recommend health checkup and ensure access to medical facilities. "
                "Consider flexible attendance policy if needed."
            ),
            "icon": "💊"
        })
    
    # ================================================================
    # RULE 9: HIGHER EDUCATION ASPIRATION CHECK
    # higher: wants to pursue higher education (yes/no)
    # ================================================================
    higher = student_data.get("higher", 1)
    
    if higher in [0, "no"] and risk_level != "Low Risk":
        recommendations.append({
            "category": "Motivation",
            "priority": "Important",
            "recommendation": (
                "Student does not aspire for higher education. "
                "Lack of long-term academic goals affects motivation. "
                "Arrange career counseling sessions. "
                "Connect with successful alumni for mentoring."
            ),
            "icon": "🎓"
        })
    
    # ================================================================
    # RULE 10: INTERNET ACCESS
    # internet: has internet at home (yes/no)
    # ================================================================
    internet = student_data.get("internet", 1)
    
    if internet in [0, "no"] and risk_level in ["High Risk", "Medium Risk"]:
        recommendations.append({
            "category": "Resources",
            "priority": "Suggested",
            "recommendation": (
                "Student lacks internet access at home. "
                "Provide printed study materials. "
                "Ensure access to school library and computer lab. "
                "Consider offline learning resources."
            ),
            "icon": "🌐"
        })
    
    # ================================================================
    # GENERAL RISK-BASED SUMMARY RECOMMENDATION
    # Always add a summary based on overall risk level
    # ================================================================
    if risk_level == "High Risk":
        recommendations.append({
            "category": "Overall Assessment",
            "priority": "Critical",
            "recommendation": (
                f"🔴 HIGH RISK ALERT: Student has only {pass_probability:.0%} "
                "probability of passing. Immediate comprehensive intervention required. "
                "Schedule emergency faculty review meeting. "
                "Create a personalized improvement plan with weekly milestones."
            ),
            "icon": "🚨"
        })
    elif risk_level == "Medium Risk":
        recommendations.append({
            "category": "Overall Assessment",
            "priority": "Important",
            "recommendation": (
                f"🟡 MEDIUM RISK: Student has {pass_probability:.0%} "
                "probability of passing. Close monitoring recommended. "
                "Set up bi-weekly check-ins. "
                "Encourage participation in study groups."
            ),
            "icon": "⚠️"
        })
    else:
        recommendations.append({
            "category": "Overall Assessment",
            "priority": "Suggested",
            "recommendation": (
                f"🟢 LOW RISK: Student has {pass_probability:.0%} "
                "probability of passing. Continue current performance. "
                "Encourage academic excellence and leadership roles. "
                "Consider advanced placement opportunities."
            ),
            "icon": "✅"
        })
    
    # Sort recommendations by priority (Critical > Important > Suggested)
    priority_order = {"Critical": 0, "Important": 1, "Suggested": 2}
    recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
    
    return recommendations


def get_risk_summary(recommendations):
    """
    Generate a brief summary of the recommendations.
    
    Counts the number of recommendations by priority level
    to give faculty a quick overview.
    
    Args:
        recommendations (list): List of recommendation dicts
    
    Returns:
        dict: Summary with counts per priority level
    """
    summary = {
        "total": len(recommendations),
        "critical": sum(1 for r in recommendations if r["priority"] == "Critical"),
        "important": sum(1 for r in recommendations if r["priority"] == "Important"),
        "suggested": sum(1 for r in recommendations if r["priority"] == "Suggested"),
    }
    return summary
