import streamlit as st
import pandas as pd
import time
import sqlite3

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from PIL import Image
import pytesseract
from PyPDF2 import PdfReader


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


st.set_page_config(
    page_title="AI Hospital Routing System",
    page_icon="🏥",
    layout="wide"
)

# =========================================================
# ADVANCED CUSTOM UI
# =========================================================

st.markdown("""
<style>

/* Main App Background */
.stApp {
    background: linear-gradient(to right, #eef2f3, #dfe9f3);
}

/* Hide Streamlit Header */
header {
    visibility: hidden;
}

/* Main Title */
.main-title {
    font-size: 45px;
    font-weight: bold;
    text-align: center;
    color: #0083B0;
    margin-top: -30px;
}

.sub-title {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}

/* Upload Box */
.upload-box {
    background-color: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

/* Result Box */
.result-box {
    background: linear-gradient(to right, #ffffff, #f7fbff);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.12);
    margin-top: 20px;
    border-left: 10px solid #0083B0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #0083B0, #005f73);
    color: white;
}

/* Sidebar Text */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 15px;
    background: linear-gradient(to right, #0083B0, #00B4DB);
    color: white;
    border: none;
    font-size: 20px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
    background: linear-gradient(to right, #005f73, #0083B0);
    color: white;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background-color: white;
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}

/* Text Area */
textarea {
    border-radius: 12px !important;
}

/* Footer */
.footer {
    text-align: center;
    color: gray;
    padding-top: 30px;
    font-size: 15px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🏥 Hospital AI")

    st.markdown("---")

    st.success("🟢 AI Routing Active")

    st.markdown("""
    ## 📂 Supported Files

    ✅ TXT Documents  
    ✅ PDF Reports    
    ✅ Prescriptions  
    ✅ Lab Reports  
    ✅ Discharge Files  

    """)

    st.markdown("---")

    st.info("⚡ Powered by NLP + OCR + Machine Learning")

    st.markdown("---")

    st.markdown("""
    ## 🧠 AI Departments

    🚨 Emergency  
    📋 Insurance  
    📁 Admin  
    💵 Billing  
    💊 Pharmacy  
    🩸 Lab Reports  
    🛏️ Discharge  
    """)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div style='
    text-align:center;
    padding:30px;
    border-radius:20px;
    background:linear-gradient(to right, #11998e, #38ef7d);
    box-shadow:0px 8px 25px rgba(0,0,0,0.15);
    margin-bottom:25px;
'>

<h1 style='
    color:white;
    font-size:52px;
    margin-bottom:12px;
    font-weight:800;
    letter-spacing:1px;
'>
🏥 AI Hospital Intelligent Routing System
</h1>

<p style='
    color:#f4fffd;
    font-size:22px;
    margin-top:0px;
    font-weight:500;
'>
⚡ Smart Medical Document Classification using OCR + NLP + Machine Learning
</p>

</div>
""", unsafe_allow_html=True)

st.markdown("---")



#  EMERGENCY MEDICAL DATASET 🚨

emergency_data = [

    # CARDIAC EMERGENCIES ❤️
    "Patient experiencing sudden crushing chest pain radiating to left arm, suspect acute myocardial infarction.",
    "Code Blue initiated in emergency ward, patient found pulseless and apneic.",
    "Severe bradycardia with loss of consciousness, prepare atropine and external pacing.",
    "Patient in ventricular tachycardia with unstable blood pressure, prepare synchronized cardioversion.",
    "Acute congestive heart failure with pulmonary edema, administer IV diuretics immediately.",
    "Patient complaining of severe palpitations and dizziness, ECG indicates atrial fibrillation with RVR.",
    "Massive hypertensive emergency with BP 240 over 130 and altered mental status.",
    "Sudden cardiac arrest witnessed in waiting area, initiate CPR immediately.",
    "Chest tightness with diaphoresis and elevated troponin levels, activate STEMI protocol.",
    "Patient collapsing repeatedly with severe hypotension and weak pulse, suspect cardiogenic shock.",

    # STROKE & NEUROLOGICAL 🧠
    "Patient exhibiting facial droop, slurred speech, and unilateral weakness, activate stroke alert.",
    "Sudden onset severe thunderclap headache with vomiting, suspect subarachnoid hemorrhage.",
    "Patient actively seizing for over 10 minutes, initiate status epilepticus protocol.",
    "Loss of consciousness following severe head injury during road traffic accident.",
    "Confused patient with unequal pupils and declining Glasgow Coma Scale.",
    "Acute ischemic stroke symptoms started within one hour, prep for thrombolysis.",
    "Patient presenting with severe neck stiffness, fever, and photophobia, suspect meningitis.",
    "Traumatic brain injury with active scalp bleeding and altered orientation.",
    "Post-trauma patient becoming unresponsive with suspected intracranial hemorrhage.",
    "Severe vertigo with ataxia and blurred vision, rule out posterior circulation stroke.",

    # TRAUMA & ACCIDENTS 🚑
    "Polytrauma patient involved in high-speed collision with suspected pelvic fracture.",
    "Gunshot wound to chest with decreased breath sounds, suspect tension pneumothorax.",
    "Open femur fracture with heavy arterial bleeding after industrial accident.",
    "Patient trapped under collapsed structure with crush injury to both lower limbs.",
    "Deep stab wound to abdomen with signs of hypovolemic shock.",
    "Motorcycle accident victim with extensive road rash and possible spinal injury.",
    "Fall from rooftop causing multiple rib fractures and respiratory compromise.",
    "Severe facial trauma with uncontrolled bleeding after physical assault.",
    "Construction worker electrocuted on site, currently unconscious and pulseless.",
    "Patient with traumatic amputation of fingers requiring emergency surgery.",

    # RESPIRATORY EMERGENCIES 🌬️
    "Acute respiratory distress with oxygen saturation falling rapidly below 75 percent.",
    "Patient wheezing severely with cyanosis, severe asthma exacerbation suspected.",
    "Anaphylactic reaction after bee sting causing airway swelling and stridor.",
    "Severe COPD exacerbation with carbon dioxide retention and confusion.",
    "Pulmonary embolism suspected due to sudden chest pain and shortness of breath.",
    "Near drowning victim presenting with frothy sputum and respiratory failure.",
    "Smoke inhalation injury after house fire, airway burns suspected.",
    "Patient coughing pink frothy sputum, possible acute pulmonary edema.",
    "Respiratory arrest following opioid overdose, administer naloxone immediately.",
    "Airway obstruction due to foreign body aspiration, initiate Heimlich maneuver.",

    # SEPSIS & INFECTION 🦠
    "Patient febrile and hypotensive with elevated lactate, septic shock suspected.",
    "Rapidly spreading cellulitis with high fever and altered mental status.",
    "Severe dehydration and septicemia following untreated gastrointestinal infection.",
    "Immunocompromised patient with neutropenic fever requiring isolation precautions.",
    "Suspected meningococcal septicemia with purpuric rash and shock.",
    "Patient presenting with toxic shock syndrome after postoperative infection.",
    "Necrotizing fasciitis suspected due to rapidly progressing soft tissue infection.",
    "COVID-positive patient deteriorating with acute respiratory failure.",
    "High-grade fever with confusion and low urine output, severe sepsis protocol activated.",
    "Patient with infected diabetic foot ulcer developing systemic inflammatory response.",

    # PEDIATRIC EMERGENCIES 👶
    "Infant presenting with febrile seizures and persistent unresponsiveness.",
    "Child accidentally ingested household bleach, poison control notified immediately.",
    "Pediatric patient with severe dehydration from acute gastroenteritis.",
    "Newborn in respiratory distress with low oxygen saturation levels.",
    "Child with severe allergic reaction after vaccination requiring epinephrine.",
    "Pediatric trauma patient hit by vehicle while crossing road.",
    "Toddler choking on coin with complete airway obstruction.",
    "Child presenting with diabetic ketoacidosis and deep labored breathing.",
    "High fever with stiff neck in pediatric patient, rule out meningitis.",
    "Unresponsive infant brought to emergency room after near drowning incident.",

    # OBSTETRIC & GYNECOLOGICAL 🤰
    "Pregnant patient with heavy vaginal bleeding and severe abdominal pain.",
    "Emergency C-section required due to fetal distress and maternal hypertension.",
    "Eclamptic seizure observed in third trimester pregnant patient.",
    "Postpartum hemorrhage with rapid blood loss after delivery.",
    "Ruptured ectopic pregnancy suspected with severe hypotension.",
    "Placental abruption causing fetal heart rate abnormalities.",
    "Pregnant trauma patient after road traffic accident requiring fetal monitoring.",
    "Severe preeclampsia with visual disturbances and elevated blood pressure.",
    "Umbilical cord prolapse detected during labor, immediate intervention required.",
    "Maternal collapse during labor with suspected amniotic fluid embolism.",

    # GASTROINTESTINAL & METABOLIC 🍽️
    "Patient vomiting large amounts of blood, suspect upper GI bleed.",
    "Acute abdomen with rebound tenderness and guarding, possible perforation.",
    "Diabetic patient unconscious with blood sugar above 600 mg per dL.",
    "Severe hypoglycemia causing seizures and altered mental status.",
    "Acute pancreatitis with severe epigastric pain radiating to back.",
    "Liver failure patient presenting with hepatic encephalopathy and confusion.",
    "Bowel obstruction suspected due to abdominal distension and vomiting.",
    "Massive diarrhea causing electrolyte imbalance and circulatory collapse.",
    "Patient with severe dehydration and acute kidney injury.",
    "Suspected ruptured abdominal aortic aneurysm with sudden severe back pain.",

    # TOXICOLOGY & OVERDOSE ☠️
    "Intentional drug overdose with decreased respiratory effort and pinpoint pupils.",
    "Carbon monoxide poisoning suspected after exposure in enclosed garage.",
    "Organophosphate poisoning with excessive salivation and muscle twitching.",
    "Alcohol intoxication with aspiration risk and altered consciousness.",
    "Unknown substance overdose requiring toxicology screening immediately.",
    "Snake bite victim developing neurotoxic symptoms and respiratory weakness.",
    "Scorpion sting causing severe autonomic storm and hypertension.",
    "Methamphetamine overdose with hyperthermia and violent agitation.",
    "Patient exposed to industrial chemical spill with respiratory burns.",
    "Opioid overdose reversed temporarily with naloxone but patient remains unstable.",

    # ENVIRONMENTAL & DISASTER 🌪️
    "Heat stroke patient with core temperature exceeding 41 degrees Celsius.",
    "Severe hypothermia after prolonged exposure to freezing temperatures.",
    "Earthquake victim trapped under debris with crush syndrome.",
    "Flood rescue patient presenting with aspiration pneumonia.",
    "Lightning strike victim unconscious with cardiac arrhythmia.",
    "Burn victim from gas explosion with airway involvement.",
    "Mass casualty incident declared after train derailment.",
    "Chemical factory explosion resulting in multiple inhalation injuries.",
    "Wildfire evacuation patient suffering severe smoke inhalation.",
    "Radiation exposure suspected following industrial accident.",

    # ICU & CRITICAL CARE 🏥
    "Patient requiring vasopressor support due to refractory septic shock.",
    "Multi-organ failure developing despite aggressive fluid resuscitation.",
    "Rapid decline in oxygenation requiring mechanical ventilation.",
    "Continuous renal replacement therapy initiated for acute renal failure.",
    "ICU patient becoming suddenly hypotensive with absent peripheral pulses.",
    "Cardiac tamponade suspected due to muffled heart sounds and hypotension.",
    "Sudden desaturation in ventilated patient, check for pneumothorax immediately.",
    "Massive transfusion protocol activated after traumatic hemorrhage.",
    "Patient developing disseminated intravascular coagulation in ICU.",
    "Critical care consult requested for unstable postoperative patient."
]

# INSURANCE MEDICAL DATASET 🏥💳

insurance_data = [

    # POLICY & COVERAGE 📄
    "The annual deductible for this health insurance policy is 1000 dollars.",
    "Out-of-pocket maximum has been reached for the current coverage year.",
    "Policyholder eligible for cashless hospitalization under network hospitals.",
    "Preventive health checkups are fully covered under this insurance plan.",
    "Maternity benefits included after completion of waiting period.",
    "Dental and vision services are excluded from the current insurance policy.",
    "Coverage includes emergency ambulance transportation expenses.",
    "Pre-existing diseases covered after a 24-month waiting period.",
    "Family floater insurance plan covers spouse and dependent children.",
    "Critical illness rider activated for cancer treatment coverage.",

    # CLAIM PROCESSING 🧾
    "Insurance claim has been successfully approved and processed.",
    "Claim number 784512 currently under medical review by insurer.",
    "Hospital submitted reimbursement request for inpatient treatment charges.",
    "Claim rejected due to incomplete supporting medical documentation.",
    "Insurance company requesting additional discharge summary for claim approval.",
    "Cashless claim authorization approved for planned surgery admission.",
    "Medical reimbursement amount transferred to registered bank account.",
    "Claim settlement delayed pending verification of treatment details.",
    "Patient submitted original pharmacy bills for reimbursement processing.",
    "Insurance adjuster reviewing ICU expenses before final settlement.",

    # PRIOR AUTHORIZATION ✅
    "Prior authorization required before MRI brain scan can be performed.",
    "Insurance preapproval mandatory for elective orthopedic surgery.",
    "CT scan request pending authorization from health insurance provider.",
    "Specialist consultation requires referral approval under policy terms.",
    "Insurer denied authorization for non-emergency cosmetic procedure.",
    "Pre-certification completed successfully for cardiac bypass surgery.",
    "Additional diagnostic procedures need insurer approval before admission.",
    "Emergency hospitalization exempt from prior authorization requirements.",
    "Outpatient chemotherapy sessions approved under cancer coverage plan.",
    "Insurance company requesting physician recommendation for authorization.",

    # BILLING & PAYMENTS 💵
    "Patient responsible for 20 percent coinsurance after deductible payment.",
    "Insurance copay for specialist consultation is 40 dollars per visit.",
    "Outstanding medical invoice forwarded to insurance billing department.",
    "Hospital billing statement submitted for insurance reimbursement.",
    "Premium payment overdue causing temporary policy suspension.",
    "Partial payment received from insurance provider for emergency treatment.",
    "Balance amount payable directly by patient after insurance adjustment.",
    "Electronic billing submitted successfully to insurance network.",
    "Coverage benefits exhausted for the current policy year.",
    "Medical expenses exceeding policy limit require self-payment.",

    # NETWORK & PROVIDER 🏨
    "Treatment performed at in-network hospital eligible for full coverage.",
    "Out-of-network consultation resulting in reduced reimbursement amount.",
    "Preferred provider organization includes this multispeciality hospital.",
    "Insurance benefits available only through empaneled healthcare providers.",
    "Referral required before visiting non-network specialist doctor.",
    "Cashless facility unavailable at non-participating hospitals.",
    "Insurance partner hospital verified patient eligibility successfully.",
    "Provider contract expired causing denial of direct billing request.",
    "Patient transferred to network hospital for insurance compliance.",
    "Network pharmacy claims processed instantly through insurance portal.",

    # DENIALS & APPEALS ❌
    "Claim denied due to non-covered experimental treatment procedure.",
    "Insurance company rejected reimbursement because policy had lapsed.",
    "Appeal submitted against denied hospitalization coverage decision.",
    "Claim denied for missing physician signature on discharge summary.",
    "Coverage refusal issued due to exclusion of pre-existing illness.",
    "Insurer requesting clarification regarding duplicate billing charges.",
    "Authorization denied for treatment exceeding policy limitations.",
    "Insurance appeal under review by grievance resolution department.",
    "Reimbursement claim rejected because admission was medically unnecessary.",
    "Patient notified regarding denied pharmacy expense reimbursement.",

    # FRAUD & COMPLIANCE ⚖️
    "Insurance audit initiated due to suspicious duplicate claims submission.",
    "Policy verification completed successfully through insurance compliance team.",
    "Fraud investigation opened for falsified hospitalization records.",
    "Insurer detected mismatch between diagnosis and billed procedures.",
    "Compliance review required before high-value claim settlement approval.",
    "Insurance documentation validated against hospital electronic records.",
    "Unauthorized policy usage reported by claims verification department.",
    "Medical coding discrepancy identified during insurance audit process.",
    "Duplicate reimbursement request flagged by insurance fraud detection system.",
    "Insurer requested identity verification before releasing payment.",

    # MEDICAL PROCEDURE COVERAGE 🩺
    "Cardiac bypass surgery fully covered under critical illness policy.",
    "Dialysis treatment eligible for recurring reimbursement benefits.",
    "Chemotherapy sessions approved under oncology insurance package.",
    "Joint replacement surgery subject to sublimit restrictions.",
    "Organ transplant expenses partially covered under premium plan.",
    "ICU room rent capped according to insurance policy terms.",
    "Physiotherapy sessions covered up to annual benefit limit.",
    "Mental health counseling included under employee insurance benefits.",
    "Vaccination charges reimbursable under preventive care coverage.",
    "Home healthcare nursing services approved for reimbursement.",

    # EMERGENCY INSURANCE 🚑
    "Emergency admission processed immediately under cashless insurance facility.",
    "Trauma care expenses covered without prior insurer authorization.",
    "Air ambulance charges approved for critical patient transfer.",
    "Emergency room copay waived during life-threatening condition treatment.",
    "Accidental injury claim registered under personal accident insurance.",
    "Road traffic accident hospitalization covered by insurer.",
    "Emergency surgery authorized following acute cardiac event.",
    "Critical care expenses exceeding standard policy limit approved exceptionally.",
    "Insurance hotline contacted for emergency hospitalization support.",
    "Emergency treatment reimbursement prioritized for immediate processing.",

    # CORPORATE & GROUP INSURANCE 👔
    "Employee group insurance policy renewed for upcoming financial year.",
    "Corporate health benefits include annual wellness screenings.",
    "Dependent coverage activated under employer-sponsored insurance plan.",
    "Human resources department submitted employee hospitalization claim.",
    "Group mediclaim coverage extended to retired staff members.",
    "Corporate insurance policy includes mental wellness benefits.",
    "Employee insurance enrollment pending document verification.",
    "Workplace injury compensation processed under occupational insurance.",
    "Employer health coverage terminated after employee resignation.",
    "Annual insurance premium deducted directly from payroll account."
]

# 
# HOSPITAL ADMINISTRATIVE DATASET 🏥📋

admin_data = [

    # APPOINTMENTS & SCHEDULING 📅
    "Please schedule a cardiology follow-up appointment for patient next Monday at 9 AM.",
    "Orthopedic consultation rescheduled due to doctor unavailability.",
    "Patient requested cancellation of tomorrow morning dental appointment.",
    "Emergency appointment slot reserved for walk-in trauma case.",
    "Doctor consultation timing updated in hospital scheduling system.",
    "Follow-up visit confirmed through SMS notification service.",
    "Double booking detected in outpatient appointment calendar.",
    "Online appointment successfully created for dermatology clinic.",
    "Patient added to waiting list for neurology consultation.",
    "Consultation delayed due to emergency surgery in operation theater.",

    # STAFF MANAGEMENT 👨‍⚕️
    "Dr. Priya Sharma is on emergency leave for the next three days.",
    "Night shift nursing staff assignment updated for ICU department.",
    "Staff attendance records submitted to human resources department.",
    "Temporary duty rotation implemented for emergency ward nurses.",
    "New receptionist joined front desk administrative team today.",
    "Monthly performance review meeting scheduled for medical staff.",
    "Payroll processing initiated for hospital employees and interns.",
    "Staff training session arranged for updated patient safety protocols.",
    "Hospital administrator approved overtime request for surgical team.",
    "Replacement physician assigned during senior consultant absence.",

    # PATIENT REGISTRATION 🧾
    "Update patient mobile number and residential address in EMR system.",
    "New patient registration completed successfully at reception counter.",
    "Duplicate medical record detected during registration verification.",
    "Insurance details updated in patient demographic profile.",
    "Emergency contact information modified as requested by patient.",
    "Patient ID card generated and printed successfully.",
    "Incorrect date of birth corrected in hospital database.",
    "Registration desk experiencing delay due to high patient volume.",
    "Patient consent form uploaded into digital records system.",
    "Admission paperwork completed for inpatient room allocation.",

    # ROOM & BED MANAGEMENT 🛏️
    "ICU bed availability updated in central bed management portal.",
    "Patient transferred from general ward to private deluxe room.",
    "Isolation room prepared for infectious disease admission.",
    "Hospital occupancy level exceeded 90 percent capacity today.",
    "Bed maintenance request submitted for broken adjustable cot.",
    "Discharge process initiated to free occupied ward bed.",
    "Emergency room bed assigned to trauma patient arrival.",
    "Housekeeping notified for immediate room sanitization process.",
    "Room allocation changed based on patient insurance eligibility.",
    "Neonatal ICU currently operating at full bed capacity.",

    # MEETINGS & COMMUNICATION 📢
    "Hospital board meeting scheduled regarding annual budget planning.",
    "Department heads requested to attend infection control conference.",
    "Emergency response drill planned for all medical staff tomorrow.",
    "Administrative circular issued regarding revised attendance policies.",
    "Internal memo distributed about updated patient privacy guidelines.",
    "Video conference arranged with external healthcare consultants.",
    "Quality assurance review meeting postponed until next week.",
    "Security briefing conducted for hospital emergency preparedness.",
    "New operational protocols communicated to nursing supervisors.",
    "Management requested incident reports before end of business day.",

    # BILLING & FINANCE 💰
    "Pending inpatient billing statements forwarded to finance department.",
    "Advance payment collected during hospital admission process.",
    "Refund request initiated for cancelled surgical procedure.",
    "Outstanding payment reminder sent to patient via email.",
    "Insurance billing discrepancy escalated for administrative review.",
    "Daily revenue summary generated for hospital management.",
    "Cash counter reconciliation completed successfully after shift closing.",
    "Medical invoice printed and handed over to patient attendant.",
    "Financial audit scheduled for outpatient billing division.",
    "Payment gateway issue causing delay in online transactions.",

    # MEDICAL RECORDS & DOCUMENTS 📂
    "Discharge summary uploaded into electronic medical records system.",
    "Patient requested duplicate copy of previous laboratory reports.",
    "Archived medical files retrieved for legal verification process.",
    "Consent document missing physician signature and requires correction.",
    "Scanning backlog pending in hospital records department.",
    "Medical transcription completed for orthopedic surgery notes.",
    "Confidential patient records accessed by authorized administrator.",
    "Document verification completed for insurance reimbursement claim.",
    "Old paper records digitized into hospital database system.",
    "Patient history requested by consulting specialist physician.",

    # FACILITY & MAINTENANCE 🏢
    "Air conditioning malfunction reported in pediatric ward corridor.",
    "Biomedical engineering team repairing faulty ventilator machine.",
    "Water supply maintenance scheduled during midnight hours.",
    "Power backup generator tested successfully during routine inspection.",
    "Elevator service temporarily unavailable due to technical repairs.",
    "Housekeeping staff instructed to sanitize waiting lounge immediately.",
    "Fire safety inspection completed across all hospital departments.",
    "Medical waste disposal pickup delayed by external vendor.",
    "Operation theater maintenance scheduled after final surgery.",
    "Internet connectivity issue affecting hospital management software.",

    # SECURITY & COMPLIANCE 🔐
    "Visitor entry restricted after official visiting hours closure.",
    "Security personnel alerted regarding unauthorized ward access attempt.",
    "Hospital CCTV surveillance footage requested for investigation.",
    "Compliance audit initiated for patient data privacy standards.",
    "Access card deactivated for resigned hospital employee.",
    "Security drill conducted for emergency evacuation preparedness.",
    "Incident report filed regarding missing medical equipment.",
    "Patient confidentiality breach escalated to compliance department.",
    "Visitor pass system updated with biometric authentication.",
    "Unauthorized parking vehicles removed from ambulance entry zone.",

    # INVENTORY & PROCUREMENT 📦
    "Pharmacy inventory running low on insulin medication stock.",
    "Purchase order generated for additional surgical gloves supply.",
    "Medical equipment delivery delayed due to transportation issues.",
    "Inventory audit revealed shortage of disposable syringes.",
    "Procurement team approved vendor quotation for ICU monitors.",
    "Expired medicines removed from central pharmacy storage.",
    "Stock replenishment request submitted for emergency crash carts.",
    "Laboratory reagents reordered for upcoming diagnostic workload.",
    "Supply chain interruption affecting oxygen cylinder availability.",
    "Central store updated inventory management records successfully.",

    # TRANSPORT & AMBULANCE 🚑
    "Ambulance dispatched for emergency patient transfer request.",
    "Driver schedule updated for night emergency transportation services.",
    "Patient transport delayed due to traffic congestion near hospital.",
    "Vehicle maintenance completed for mobile diagnostic van.",
    "Inter-hospital transfer arranged for critical cardiac patient.",
    "Ambulance fuel expenses submitted for reimbursement approval.",
    "Transport coordination team notified regarding organ transfer mission.",
    "Emergency vehicle GPS tracking temporarily offline.",
    "Wheelchair transport assistance requested for elderly patient.",
    "Ambulance sanitization completed after infectious disease transfer."
]

# 4. Billing Data 💵


billing_data = [

    # GENERAL HOSPITAL BILLING 🧾
    "Total hospital charges for inpatient treatment amount to 4500 dollars.",
    "Outstanding medical bill pending payment before discharge process.",
    "Invoice generated successfully for emergency room consultation.",
    "Final billing statement shared with patient attendant via email.",
    "Advance payment adjusted against total hospitalization expenses.",
    "Billing department requested immediate clearance of pending dues.",
    "Patient account updated after successful online payment transaction.",
    "Hospital invoice includes consultation, pharmacy, and laboratory charges.",
    "Detailed expense summary printed for insurance reimbursement submission.",
    "Outstanding balance remains unpaid after insurance deduction adjustment.",

    # ROOM & ADMISSION CHARGES 🛏️
    "Private room charges calculated at 300 dollars per night stay.",
    "ICU admission billing includes ventilator and monitoring expenses.",
    "General ward accommodation fees added to patient invoice.",
    "Room upgrade charges applied during hospitalization period.",
    "Bed occupancy charges revised according to updated pricing policy.",
    "Emergency admission deposit collected at registration counter.",
    "Deluxe suite charges exceed standard insurance coverage limits.",
    "Isolation ward fees included in infectious disease treatment bill.",
    "Additional nursing care charges added for ICU patient management.",
    "Hospital stay extended resulting in increased accommodation costs.",

    # LAB & DIAGNOSTIC BILLING 🧪
    "MRI brain scan billed separately under radiology department charges.",
    "Blood test and pathology expenses included in diagnostic invoice.",
    "CT scan payment pending before report release authorization.",
    "Ultrasound billing completed successfully through hospital portal.",
    "Laboratory investigation charges updated in billing software system.",
    "X-ray procedure billed under outpatient diagnostic services.",
    "Cardiac enzyme testing charges added to emergency evaluation bill.",
    "Repeated diagnostic testing increased total treatment expenses.",
    "Pathology department invoice generated for biopsy examination.",
    "Advanced imaging procedures require upfront payment confirmation.",

    # SURGERY & PROCEDURE BILLING 🔪
    "Operation theater charges included anesthesia and surgical equipment costs.",
    "Cardiac bypass surgery invoice exceeds estimated treatment package amount.",
    "Surgical advance payment required before elective procedure admission.",
    "Procedure billing updated after completion of laparoscopic surgery.",
    "Postoperative care charges added to final discharge bill.",
    "Surgeon consultation fee billed separately from hospital expenses.",
    "Emergency appendectomy charges processed under surgical billing unit.",
    "Implant costs added additionally to orthopedic surgery invoice.",
    "Minor procedure charges generated during outpatient treatment visit.",
    "Operation cancellation fee applied according to hospital policy.",

    # PHARMACY & MEDICATION 💊
    "Pharmacy invoice includes antibiotics, painkillers, and IV medications.",
    "Medicine charges updated automatically through electronic prescription system.",
    "High-cost injectable drugs significantly increased patient bill amount.",
    "Discount applied to pharmacy bill under hospital membership plan.",
    "Medication expenses pending verification by insurance provider.",
    "Additional drug administration charges added during ICU treatment.",
    "Pharmacy billing discrepancy reported by patient family member.",
    "Prescription refill charges processed at outpatient pharmacy counter.",
    "Controlled medication billing requires physician authorization approval.",
    "Medication wastage charges added during chemotherapy preparation.",

    # INSURANCE & CLAIM BILLING 🏥💰
    "Insurance claim submitted successfully for hospitalization reimbursement.",
    "Cashless billing approved under network insurance coverage policy.",
    "Insurance deduction reflected in revised patient invoice summary.",
    "Claim settlement delayed due to missing billing documentation.",
    "Patient responsible for non-covered treatment expenses after insurance adjustment.",
    "Billing dispute raised regarding denied insurance reimbursement claim.",
    "Co-payment amount payable directly by patient before discharge.",
    "Medical coding error caused temporary insurance claim rejection.",
    "Insurance preauthorization required for expensive diagnostic procedure billing.",
    "Reimbursement invoice generated according to insurer documentation guidelines.",

    # PAYMENT & TRANSACTIONS 💵
    "Online payment received successfully through hospital billing portal.",
    "Credit card transaction failed during bill payment processing.",
    "Partial payment accepted with remaining balance due next week.",
    "Cash receipt generated after successful payment at billing counter.",
    "EMI payment option available for high-value treatment expenses.",
    "Late payment penalty applied to overdue hospital invoice.",
    "Refund initiated for cancelled surgery advance deposit amount.",
    "Duplicate payment detected during financial reconciliation process.",
    "Transaction confirmation shared via SMS and registered email address.",
    "Bank transfer details provided for international patient billing.",

    # EMERGENCY & ICU BILLING 🚑
    "Emergency trauma care charges updated in critical care invoice.",
    "Ventilator support expenses billed hourly in ICU department.",
    "Crash cart usage charges included during cardiac arrest management.",
    "Emergency surgery costs exceeded initial treatment estimate significantly.",
    "Ambulance transportation fee added to emergency admission bill.",
    "Critical care specialist consultation charged separately from ICU stay.",
    "Emergency medication administration fees reflected in patient account.",
    "Life-saving intervention charges approved under emergency billing policy.",
    "Intensive monitoring costs accumulated during prolonged ICU admission.",
    "Emergency blood transfusion expenses added to trauma treatment invoice.",

    # OUTPATIENT & CONSULTATION BILLING 👨‍⚕️
    "Specialist consultation charges payable before outpatient examination.",
    "Follow-up visit billed at discounted consultation rate.",
    "Telemedicine appointment invoice generated through hospital app.",
    "Walk-in consultation fee collected at outpatient reception desk.",
    "Doctor review charges updated after extended consultation session.",
    "Dietician and physiotherapy consultation billed separately.",
    "Repeat consultation within seven days eligible for reduced fee.",
    "Outpatient procedure charges included dressing and injection expenses.",
    "Vaccination administration fee added during pediatric consultation.",
    "Dermatology laser treatment billed under cosmetic procedure category.",

    # CORPORATE & PACKAGE BILLING 📦
    "Corporate health checkup package billed to employer account directly.",
    "Annual wellness package charges processed under subscription plan.",
    "Maternity package billing includes delivery and neonatal care services.",
    "Comprehensive cardiac package invoice generated successfully.",
    "Executive health screening charges approved by company insurance policy.",
    "Package inclusions revised during extended hospitalization period.",
    "Discounted bundled pricing applied for preventive health examinations.",
    "Employee healthcare expenses billed through corporate agreement contract.",
    "Surgical package excluded additional complication management charges.",
    "Fixed-price treatment package exceeded due to ICU complications.",

    # BILLING AUDIT & COMPLIANCE ⚖️
    "Billing audit initiated due to discrepancy in patient account records.",
    "Duplicate laboratory charges removed after financial review process.",
    "Incorrect medication billing corrected by accounts department.",
    "Hospital finance team investigating disputed invoice amount.",
    "Compliance review required before finalizing high-value medical bill.",
    "Unauthorized service charges reversed after patient complaint.",
    "Medical billing software updated according to revised tax regulations.",
    "Audit trail maintained for all hospital financial transactions.",
    "Revenue cycle management team reviewing incomplete payment entries.",
    "Financial reconciliation completed successfully for discharged patient account."
]


 # PHARMACY& MEDICATION DATASET 💊🏥

pharmacy_data = [

    # PRESCRIPTIONS & DOSAGE 🩺
    "Patient prescribed Amoxicillin 500mg twice daily for bacterial infection treatment.",
    "Take Paracetamol 650mg every six hours as needed for fever management.",
    "Doctor prescribed Metformin 1000mg after meals for diabetes control.",
    "Administer Insulin injection subcutaneously before breakfast and dinner.",
    "Prescription includes Pantoprazole 40mg once daily before food intake.",
    "Azithromycin prescribed for five days to treat respiratory infection.",
    "Patient advised to continue antihypertensive medication every morning.",
    "Prescription updated with Vitamin D supplements for deficiency treatment.",
    "Oral rehydration salts recommended after every loose stool episode.",
    "Doctor instructed patient to complete full antibiotic treatment course.",

    # PHARMACY DISPENSING 💊
    "Prescription medicines dispensed successfully at outpatient pharmacy counter.",
    "Medication packet labeled with dosage instructions and warning signs.",
    "Controlled drug dispensing requires physician signature verification.",
    "Pharmacist confirmed correct dosage before releasing medications.",
    "Partial prescription dispensed due to temporary stock shortage.",
    "Medication barcode scanned successfully into pharmacy management system.",
    "Refill request approved and prepared for patient collection.",
    "Patient counseling provided regarding proper medicine administration timing.",
    "Expired medicines removed from dispensing shelf during inspection.",
    "Prescription refill denied due to expired physician authorization.",

    # STOCK & INVENTORY 📦
    "Pharmacy inventory running low on insulin vials and syringes.",
    "Emergency stock replenishment requested for adrenaline injection supply.",
    "Salbutamol inhalers currently out of stock in central pharmacy.",
    "New shipment of antibiotics received and added to inventory records.",
    "Cold chain monitoring completed for vaccine refrigerator storage.",
    "Expired chemotherapy drugs quarantined for safe disposal procedure.",
    "Pharmacy database updated after medication stock audit completion.",
    "Oxygen cylinder supply verified for emergency department pharmacy.",
    "Inventory discrepancy identified during narcotic medication count.",
    "Shortage reported for pediatric fever syrup formulations.",

    # EMERGENCY MEDICATIONS 🚑
    "Administer IV adrenaline immediately for severe anaphylactic reaction.",
    "Emergency crash cart stocked with atropine and epinephrine injections.",
    "Naloxone administered during suspected opioid overdose management.",
    "Lorazepam IV push ordered for ongoing seizure activity.",
    "Rapid sequence intubation medications prepared for critical patient.",
    "Nitroglycerin tablets given for acute chest pain symptoms.",
    "Emergency anticoagulants initiated for pulmonary embolism treatment.",
    "High-dose steroids prescribed during severe asthma exacerbation.",
    "Dopamine infusion started to maintain blood pressure support.",
    "Blood transfusion medications prepared for trauma resuscitation.",

    # CHRONIC DISEASE MEDICATIONS ❤️
    "Amlodipine prescribed daily for long-term hypertension management.",
    "Insulin dosage adjusted according to recent glucose monitoring reports.",
    "Patient continuing atorvastatin therapy for cholesterol reduction.",
    "Levothyroxine prescribed for hypothyroidism treatment maintenance.",
    "Warfarin therapy requires regular INR monitoring and dose adjustments.",
    "Bronchodilator inhaler prescribed for chronic obstructive pulmonary disease.",
    "Antiepileptic medication refill approved for seizure disorder control.",
    "Oral hypoglycemic agents continued for type 2 diabetes management.",
    "Heart failure medications optimized after cardiology consultation.",
    "Long-term antidepressant prescription renewed after psychiatric review.",

    # PEDIATRIC MEDICATIONS 👶
    "Pediatric syrup dosage calculated according to child body weight.",
    "Infant prescribed oral zinc supplementation for acute diarrhea treatment.",
    "Vaccination schedule updated in pediatric immunization record.",
    "Paracetamol drops prescribed for fever in six-month-old infant.",
    "Nebulization therapy initiated for wheezing child with asthma symptoms.",
    "Pediatric antibiotic suspension requires refrigeration after preparation.",
    "Vitamin supplements prescribed for underweight pediatric patient.",
    "Antipyretic medication administered during febrile seizure management.",
    "Oral antibiotic dose reduced according to pediatric safety guidelines.",
    "Child prescribed antihistamine syrup for allergic skin rash.",

    # ICU & HOSPITAL MEDICATIONS 🏥
    "Continuous IV sedation maintained for mechanically ventilated patient.",
    "Broad-spectrum antibiotics initiated for suspected septic shock.",
    "Electrolyte replacement therapy ordered for ICU patient imbalance.",
    "Intravenous vasopressor infusion titrated according to blood pressure.",
    "Postoperative pain management includes opioid infusion therapy.",
    "Anticoagulant prophylaxis started for immobilized ICU patient.",
    "Central pharmacy dispatched emergency medications to intensive care unit.",
    "Critical care patient receiving total parenteral nutrition support.",
    "Insulin infusion protocol activated for diabetic ketoacidosis treatment.",
    "Medication administration delayed due to pharmacy verification process.",

    # PHARMACY OPERATIONS 🧾
    "Electronic prescription transmitted directly to hospital pharmacy system.",
    "Pharmacist contacted physician regarding potential drug interaction warning.",
    "Medication administration record updated after evening drug rounds.",
    "Pharmacy billing generated for outpatient prescription medicines.",
    "Drug utilization review completed for inpatient treatment regimen.",
    "Hospital pharmacy closed temporarily for scheduled maintenance activity.",
    "Automated dispensing cabinet refilled during overnight pharmacy shift.",
    "Medication reconciliation completed during patient discharge planning.",
    "Prescription scanning system temporarily unavailable due to technical issue.",
    "Pharmacy staff meeting scheduled regarding medication safety procedures.",

    # ALLERGIES & DRUG REACTIONS ⚠️
    "Patient developed allergic rash after penicillin administration.",
    "Severe drug reaction reported following intravenous antibiotic infusion.",
    "Medication allergy alert triggered for sulfa-containing prescription.",
    "Patient advised to avoid NSAIDs due to gastric ulcer history.",
    "Anaphylaxis suspected after administration of contrast medication.",
    "Drug interaction warning issued for combined anticoagulant therapy.",
    "Patient experienced dizziness after antihypertensive dose increase.",
    "Adverse drug event documented in pharmacovigilance reporting system.",
    "Opioid medication discontinued due to respiratory depression symptoms.",
    "Allergy bracelet updated with newly identified medication reaction.",

    # SPECIALIZED MEDICATIONS 🧬
    "Chemotherapy drugs prepared under sterile oncology pharmacy conditions.",
    "Immunosuppressant therapy initiated after organ transplantation surgery.",
    "Antiretroviral medications prescribed for HIV infection management.",
    "Biologic injection approved for autoimmune arthritis treatment.",
    "Dialysis patient prescribed erythropoietin injections weekly.",
    "Cancer pain management includes controlled morphine administration.",
    "Antiviral medications started during acute hepatitis infection treatment.",
    "Targeted therapy drugs ordered for metastatic lung cancer patient.",
    "Radiology contrast agents supplied for MRI diagnostic procedure.",
    "Rare disease medication sourced through specialty pharmacy provider.",

    # OVER-THE-COUNTER & GENERAL MEDICINES 🛒
    "Cough syrup recommended for nighttime throat irritation relief.",
    "Antacid tablets suggested for mild gastric acidity symptoms.",
    "Multivitamin supplements available without prescription requirement.",
    "Topical antiseptic cream prescribed for minor skin infection.",
    "Pain relief gel advised for muscle strain and joint discomfort.",
    "Oral antihistamines recommended for seasonal allergy management.",
    "Hydration salts suggested during viral fever recovery phase.",
    "Herbal cough lozenges purchased from outpatient pharmacy section.",
    "Nasal decongestant spray dispensed for sinus congestion treatment.",
    "Eye lubricant drops recommended for dry eye irritation symptoms."
]



# LAB REPORTS & DIAGNOSTIC DATASET 🩸🔬

lab_data = [

    # BLOOD TESTS 🩸
    "Hemoglobin levels are significantly low indicating moderate iron deficiency anemia.",
    "White blood cell count elevated suggesting active bacterial infection.",
    "Platelet count critically reduced requiring urgent hematology evaluation.",
    "Fasting blood glucose measured at 240 mg per dL indicating uncontrolled diabetes.",
    "HbA1c levels elevated above target range over previous three months.",
    "Serum creatinine increased suggesting impaired kidney function.",
    "Liver enzymes ALT and AST markedly elevated in blood analysis.",
    "Cholesterol profile shows elevated LDL and triglyceride levels.",
    "Potassium levels critically high indicating severe electrolyte imbalance.",
    "C-reactive protein elevated suggesting systemic inflammatory response.",

    # URINE ANALYSIS 🚽
    "Urine culture positive for Escherichia coli indicating urinary tract infection.",
    "Proteinuria detected in urine sample suggesting possible renal disease.",
    "Urinalysis reveals ketone bodies consistent with diabetic ketoacidosis.",
    "Microscopic hematuria observed during routine urine examination.",
    "Urine pregnancy test returned positive result successfully.",
    "Bacteria and pus cells detected in urine microscopy analysis.",
    "Urinary pH levels within normal physiological range.",
    "Glucose detected in urine sample during diabetic screening.",
    "Urine toxicology screen positive for opioid substances.",
    "Specific gravity reduced indicating diluted urine concentration.",

    # MRI REPORTS 🧠
    "MRI brain scan reveals acute ischemic infarct in frontal lobe region.",
    "Lumbar spine MRI demonstrates L4-L5 intervertebral disc herniation.",
    "MRI knee shows anterior cruciate ligament tear with joint effusion.",
    "Contrast MRI indicates suspicious intracranial mass lesion.",
    "Cervical spine MRI reveals spinal canal stenosis and nerve compression.",
    "MRI abdomen demonstrates fatty liver changes without focal lesions.",
    "Brain MRI normal with no evidence of acute hemorrhage.",
    "Pelvic MRI suggests ovarian cyst requiring gynecological evaluation.",
    "MRI shoulder reveals rotator cuff tendon partial thickness tear.",
    "Multiple sclerosis plaques identified on MRI neuroimaging study.",

    # CT SCAN REPORTS 🖥️
    "CT chest scan demonstrates bilateral pneumonia with pleural effusion.",
    "Abdominal CT reveals acute appendicitis with surrounding inflammation.",
    "CT brain shows intracranial hemorrhage following traumatic injury.",
    "Pulmonary angiography confirms presence of pulmonary embolism.",
    "CT abdomen indicates bowel obstruction with dilated intestinal loops.",
    "Facial bone CT demonstrates multiple orbital fractures.",
    "CT coronary angiogram reveals severe arterial stenosis.",
    "Chest CT negative for pneumothorax after trauma assessment.",
    "Renal CT scan identifies obstructive ureteric calculus.",
    "CT pelvis suggests pelvic hematoma after motor vehicle accident.",

    # X-RAY REPORTS 🦴
    "Chest X-ray reveals right lower lobe consolidation consistent with pneumonia.",
    "X-ray wrist demonstrates distal radius fracture displacement.",
    "Portable chest radiograph confirms endotracheal tube placement.",
    "Pelvic X-ray indicates fracture involving left hip joint.",
    "Spine radiograph shows degenerative spondylotic changes.",
    "X-ray findings suggest mild osteoarthritis of knee joint.",
    "No acute cardiopulmonary abnormality detected on chest radiograph.",
    "Rib X-ray reveals multiple fractures after blunt chest trauma.",
    "Abdominal X-ray demonstrates air-fluid levels indicating obstruction.",
    "Follow-up radiograph confirms satisfactory fracture alignment.",

    # BIOPSY & PATHOLOGY 🔬
    "Biopsy findings confirm benign fibroadenoma without malignant transformation.",
    "Histopathology reveals invasive ductal carcinoma breast tissue.",
    "Cervical biopsy negative for dysplasia or malignancy evidence.",
    "Liver biopsy demonstrates chronic inflammatory hepatitis changes.",
    "Bone marrow examination indicates acute leukemia infiltration.",
    "Pathology report confirms squamous cell carcinoma diagnosis.",
    "Tissue specimen shows granulomatous inflammation suggestive of tuberculosis.",
    "Endometrial biopsy reveals hyperplasia without atypical cellular features.",
    "Colon biopsy positive for ulcerative colitis inflammatory changes.",
    "Lymph node pathology indicates metastatic malignant involvement.",

    # MICROBIOLOGY & CULTURE 🦠
    "Blood cultures positive for methicillin-resistant Staphylococcus aureus.",
    "Sputum culture demonstrates multidrug resistant bacterial growth.",
    "COVID-19 RT-PCR test returned positive detection result.",
    "Malaria parasite identified in peripheral blood smear examination.",
    "Tuberculosis GeneXpert test positive for Mycobacterium tuberculosis.",
    "Fungal culture reveals Candida species overgrowth infection.",
    "Dengue NS1 antigen test reactive in serum sample.",
    "Throat swab culture negative for streptococcal infection.",
    "Hepatitis B surface antigen positive in screening analysis.",
    "CSF culture sterile with no bacterial growth detected.",

    # CARDIAC & ECG ❤️
    "Electrocardiogram demonstrates ST segment elevation myocardial infarction.",
    "Echocardiogram reveals reduced left ventricular ejection fraction.",
    "Holter monitoring detected intermittent atrial fibrillation episodes.",
    "Cardiac troponin levels elevated consistent with myocardial injury.",
    "Stress test positive for inducible myocardial ischemia.",
    "ECG findings suggest ventricular hypertrophy and arrhythmia.",
    "Echocardiography shows moderate mitral valve regurgitation.",
    "Cardiac biomarkers remain within normal reference limits.",
    "Telemetry monitoring captured episodes of ventricular tachycardia.",
    "Pericardial effusion identified during bedside echocardiographic assessment.",

    # HORMONE & ENDOCRINE 🧬
    "Thyroid stimulating hormone elevated indicating hypothyroidism.",
    "Serum cortisol levels abnormally low during endocrine evaluation.",
    "Vitamin D deficiency confirmed through laboratory analysis.",
    "Testosterone levels reduced below expected age-specific range.",
    "Parathyroid hormone elevated suggesting calcium metabolism disorder.",
    "Insulin levels elevated indicating insulin resistance syndrome.",
    "Prolactin concentration increased requiring pituitary imaging evaluation.",
    "FSH and LH levels consistent with menopausal hormonal changes.",
    "Thyroid profile within normal laboratory reference values.",
    "Growth hormone deficiency suspected based on endocrine testing.",

    # EMERGENCY & CRITICAL LABS 🚑
    "Arterial blood gas analysis reveals severe metabolic acidosis.",
    "Lactate levels critically elevated indicating tissue hypoperfusion.",
    "D-dimer markedly increased raising suspicion for thromboembolism.",
    "Coagulation profile abnormal with prolonged INR and PT values.",
    "Troponin levels rising rapidly during serial cardiac testing.",
    "Septic screening markers elevated in critically ill patient.",
    "Electrolyte panel indicates life-threatening hyponatremia condition.",
    "Crossmatch completed successfully for emergency blood transfusion.",
    "Toxicology screen positive for benzodiazepine overdose exposure.",
    "Emergency glucose testing confirms severe hypoglycemia episode.",

    # GENERAL HEALTH SCREENING 🏥
    "Routine health checkup results within normal laboratory ranges.",
    "Annual preventive screening indicates borderline cholesterol elevation.",
    "Complete blood count demonstrates mild viral infection pattern.",
    "Kidney function tests normal with stable electrolyte balance.",
    "Lipid profile improved following dietary and lifestyle modifications.",
    "Routine liver function screening reveals no significant abnormalities.",
    "Bone density scan suggests early osteopenia changes.",
    "Vision screening report indicates mild refractive error condition.",
    "Pulmonary function test consistent with mild obstructive airway disease.",
    "Screening mammogram shows no suspicious breast abnormalities."
]
#  DISCHARGE SUMMARIES DATASET 🛏️🏥

discharge_data = [

    # GENERAL DISCHARGE NOTES 📄
    "Patient discharged in stable condition with no active complaints.",
    "Condition improved significantly after treatment and observation period.",
    "Vitals stable at discharge and patient is hemodynamically stable.",
    "Patient discharged with advice for adequate rest and hydration.",
    "No complications observed during hospital stay, safe for discharge.",
    "Patient discharged with caregiver support and home care instructions.",
    "Clinical condition resolved and patient cleared for discharge.",
    "Discharge approved after satisfactory recovery progress evaluation.",
    "Patient left hospital in good general condition.",
    "Stable at discharge with instructions for outpatient follow-up care.",

    # SURGICAL DISCHARGE 🩺🔪
    "Patient admitted for laparoscopic appendectomy, procedure completed successfully.",
    "Post-operative recovery uneventful following gallbladder removal surgery.",
    "Sutures placed during surgery to be removed after 7 to 10 days.",
    "Wound healing satisfactory after orthopedic fracture fixation procedure.",
    "Discharged after successful cesarean section with healthy mother and baby.",
    "Post-surgical infection not observed during hospital stay.",
    "Patient advised to continue wound dressing at home daily.",
    "Knee replacement surgery completed with good post-operative mobility.",
    "Discharge after hernia repair surgery with activity restrictions advised.",
    "Follow-up scheduled after surgical biopsy and histopathology review.",

    # MEDICAL TREATMENT DISCHARGE 💊
    "Final diagnosis acute bronchitis, antibiotics prescribed for five days.",
    "Pneumonia treated successfully with IV antibiotics and supportive care.",
    "Asthma exacerbation controlled and inhaler prescribed at discharge.",
    "Patient treated for urinary tract infection and discharged on oral antibiotics.",
    "Diabetes stabilized during admission and medication adjusted at discharge.",
    "Hypertension controlled with new antihypertensive regimen prescribed.",
    "Gastritis symptoms improved after proton pump inhibitor therapy.",
    "Viral fever resolved with supportive treatment and hydration therapy.",
    "Migraine symptoms managed and prophylactic medication prescribed.",
    "Skin infection treated successfully with topical and oral antibiotics.",

    # FOLLOW-UP INSTRUCTIONS 📅
    "Patient advised to follow up with physician after one week.",
    "Next outpatient visit scheduled for wound review and suture removal.",
    "Follow-up echocardiogram recommended after cardiac treatment.",
    "Patient instructed to return if symptoms worsen or recur.",
    "Routine check-up advised in two weeks for reassessment.",
    "Specialist consultation required for further evaluation post-discharge.",
    "Follow-up blood tests scheduled for monitoring recovery progress.",
    "Physiotherapy sessions recommended after orthopedic discharge.",
    "Repeat imaging advised after one month for condition review.",
    "Patient instructed to visit emergency if severe symptoms appear.",

    # MEDICATION AT DISCHARGE 💊
    "Discharged with prescription for oral antibiotics and pain relief medication.",
    "Medication regimen adjusted and explained to patient at discharge.",
    "Patient advised to continue antihypertensive drugs regularly.",
    "Insulin dosage modified and home monitoring instructions provided.",
    "Analgesics prescribed for post-operative pain management at home.",
    "Steroid tapering schedule explained before discharge.",
    "Anticoagulant therapy continued with INR monitoring instructions.",
    "Vitamin supplements prescribed for recovery support.",
    "Inhaler technique demonstrated before discharge for asthma control.",
    "Complete medication list provided to patient and caregiver.",

    # ACTIVITY RESTRICTIONS 🚫
    "Patient advised to avoid heavy lifting for three months post-surgery.",
    "Strict bed rest recommended for initial recovery phase.",
    "Avoid strenuous physical activity until follow-up review.",
    "No driving advised for at least two weeks after discharge.",
    "Restricted mobility instructions given after orthopedic procedure.",
    "Avoid exposure to infectious environments during recovery period.",
    "Dietary restrictions advised for gastrointestinal recovery.",
    "Patient instructed to avoid alcohol consumption during medication course.",
    "Limited walking recommended during early post-operative phase.",
    "Work resumption advised only after physician clearance.",

    # OBSTETRIC DISCHARGE 🤰
    "Mother and newborn discharged in stable condition after normal delivery.",
    "Post-cesarean recovery uneventful with healthy infant outcome.",
    "Breastfeeding guidance provided before discharge.",
    "Postpartum care instructions explained to patient and family.",
    "No complications observed during labor and delivery process.",
    "Mother advised rest and nutritional support during recovery.",
    "Follow-up obstetric checkup scheduled in six weeks.",
    "Neonatal screening completed prior to discharge.",
    "Eclampsia resolved and patient stabilized before discharge.",
    "Patient counseled on contraception options post-delivery.",

    # CRITICAL CARE DISCHARGE 🏥
    "Patient transferred from ICU to general ward before discharge.",
    "Mechanical ventilation successfully weaned prior to discharge.",
    "Septic shock resolved after intensive antibiotic therapy.",
    "Cardiac status stabilized following intensive monitoring.",
    "Multi-organ function improved significantly during ICU stay.",
    "Patient discharged after successful recovery from critical illness.",
    "Continuous monitoring discontinued after stable vitals achieved.",
    "No further ICU care required at time of discharge.",
    "Vasopressor support discontinued prior to discharge.",
    "Neurological status improved after intensive care management.",

    # EMERGENCY DISCHARGE 🚑
    "Emergency case stabilized and discharged after observation period.",
    "Trauma patient discharged after ruling out internal injuries.",
    "Chest pain evaluated and cardiac emergency excluded.",
    "Patient discharged after negative imaging and lab results.",
    "Minor injury treated and patient sent home same day.",
    "Allergic reaction resolved after emergency treatment and observation.",
    "Seizure patient stabilized and discharged with medication.",
    "Poisoning case managed and patient discharged in stable condition.",
    "Fracture immobilized and patient discharged with orthopedic advice.",
    "Emergency admission resolved without surgical intervention.",

    # INSTRUCTIONS & EDUCATION 📚
    "Patient educated on medication adherence and lifestyle modification.",
    "Dietary advice provided for diabetes and cholesterol control.",
    "Wound care instructions demonstrated to patient caregiver.",
    "Signs of complication explained before hospital discharge.",
    "Physiotherapy exercises taught for home recovery program.",
    "Smoking cessation advice given for respiratory health improvement.",
    "Hydration and rest emphasized for viral infection recovery.",
    "Warning symptoms explained for immediate hospital return.",
    "Follow-up compliance importance explained to patient.",
    "Family counseling provided regarding home care responsibilities."
]


# =========================================================
# DATABASE SETUP
# =========================================================



conn = sqlite3.connect("hospital_docs.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    text TEXT,
    category TEXT
)
""")

conn.commit()

# =========================================================
# DATAFRAME
# =========================================================

all_texts = []
all_labels = []

category_mapping = {
    "Emergency 🚨": emergency_data,
    "Insurance 📋": insurance_data,
    "Admin 📁": admin_data,
    "Billing 💵": billing_data,
    "Pharmacy 💊": pharmacy_data,
    "Lab Reports 🩸": lab_data,
    "Discharge Summaries 🛏️": discharge_data
}

for label, texts in category_mapping.items():

    all_texts.extend(texts)

    all_labels.extend([label] * len(texts))

df = pd.DataFrame({
    "text": all_texts,
    "category": all_labels
})

# =========================================================
# MODEL TRAINING
# =========================================================

@st.cache_resource
def train_model():

    model = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB()
    )

    model.fit(df["text"], df["category"])

    return model

model = train_model()

# =========================================================
# FILE UPLOAD
# =========================================================

st.write("## 📄 Upload Medical Document")

uploaded_file = st.file_uploader(
    "Upload TXT / PDF / Image",
    type=["txt", "pdf", "png", "jpg", "jpeg"]
)

user_input = ""

# =========================================================
# FILE PROCESSING
# =========================================================

if uploaded_file is not None:

    # TEXT FILE
    if uploaded_file.type == "text/plain":

        user_input = uploaded_file.read().decode("utf-8")

        st.success("✅ Text file uploaded successfully!")

    # IMAGE FILE
    elif uploaded_file.type.startswith("image"):

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Medical Image",
            use_container_width=True
        )

        try:

            user_input = pytesseract.image_to_string(image)

            st.success("✅ Text extracted from image successfully!")

        except:

            st.error("❌ Tesseract OCR is not installed!")

    # PDF FILE
    elif uploaded_file.type == "application/pdf":

        pdf_reader = PdfReader(uploaded_file)

        for page in pdf_reader.pages:

            text = page.extract_text()

            if text:
                user_input += text

        st.success("✅ PDF text extracted successfully!")

    # SHOW EXTRACTED TEXT
    st.text_area(
        "Extracted Text",
        value=user_input,
        height=200
    )

else:

    user_input = st.text_area(
        "Or Paste Medical Document Text Here",
        height=180
    )

if st.button(
    "🔍 Categorize Document",
    use_container_width=True
):

    if user_input.strip() != "":

        with st.spinner("🧠 AI analyzing hospital document..."):

            time.sleep(1)

            # =================================================
            # PREDICTION
            # =================================================

            prediction = model.predict([user_input])[0]

            prediction_proba = model.predict_proba([user_input])[0]

            confidence = max(prediction_proba) * 100

            st.success("✅ Analysis Complete!")

            # =================================================
            # SAVE TO DATABASE
            # =================================================

            cursor.execute(
                "INSERT INTO predictions VALUES (?, ?)",
                (user_input, prediction)
            )

            conn.commit()

            # =================================================
            # METRICS
            # =================================================

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Predicted Department",
                    prediction
                )

            with col2:

                st.metric(
                    "Confidence Score",
                    f"{confidence:.2f}%"
                )

            # =================================================
            # CONFIDENCE BAR
            # =================================================

            st.progress(confidence / 100)

            
            # =================================================
            # ROUTING ALERTS
            # =================================================

            if "Emergency" in prediction:

                st.error(
                    "🚨 Routing to Emergency Department"
                )

            elif "Billing" in prediction:

                st.warning(
                    "💵 Routing to Billing Department"
                )

            elif "Pharmacy" in prediction:

                st.success(
                    "💊 Routing to Pharmacy"
                )

            elif "Lab Reports" in prediction:

                st.info(
                    "🩸 Routing to Pathology Lab"
                )

            elif "Insurance" in prediction:

                st.info(
                    "📋 Routing to Insurance Desk"
                )

            elif "Admin" in prediction:

                st.info(
                    "📁 Routing to Administration"
                )

            elif "Discharge" in prediction:

                st.success(
                    "🛏️ Routing to Discharge Section"
                )

            # =================================================
            # DOWNLOAD REPORT
            # =================================================

            report = f"""
AI HOSPITAL ROUTING REPORT

Predicted Department:
{prediction}

Confidence Score:
{confidence:.2f}%

--------------------------------

Document Text:

{user_input}
"""

            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name="hospital_report.txt",
                mime="text/plain",
                use_container_width=True
            )

    else:

        st.warning(
            "⚠️ Please upload a document or enter text."
        )

# =========================================================
# HISTORY SECTION
# =========================================================

st.markdown("---")

if st.checkbox("📜 Show Previous Predictions"):

    history = pd.read_sql(
        "SELECT * FROM predictions",
        conn
    )

    st.dataframe(
        history,
        use_container_width=True
    )
# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<hr>

<center>
<p style='color:gray;'>

Built with ❤️ using Streamlit | AI Healthcare Routing System

</p>
</center>
""", unsafe_allow_html=True)

# =========================================================
# CLOSE DATABASE
# =========================================================

conn.close()