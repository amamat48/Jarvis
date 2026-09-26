#include <iostream>

using namespace std;

int main() {

    const double COST_PER_PATIENT = 1375.61;
    int firstYear, secondYear;
    int year1Patients, year2Patients;

    cout << "Enter the first year: ";
    cin >> firstYear;
    cout << "Enter the number of patients we saw that year: ";
    cin >> year1Patients;
    
    cout << "Enter the second year: ";
    cin >> secondYear;
    cout << "Enter the number of patients we saw that year: ";
    cin >> year2Patients;

    double year1Charity = year1Patients * COST_PER_PATIENT;
    double year2Charity = year2Patients * COST_PER_PATIENT;

    double patientPerYearDifference = (static_cast<double>(year2Patients - year1Patients) / year1Patients) * 100;

    cout << "In year " << firstYear << ", our hospital issued " << year1Charity << " dollars of charity charges." << endl;

    cout << "In year " << secondYear << ", our hospital issued " << year2Charity << " dollars of charity charges." << endl;

    cout << "Between " << firstYear << " and " << secondYear << ", there was a " << patientPerYearDifference << "% increase in patients seen at our hospital." << endl;

}