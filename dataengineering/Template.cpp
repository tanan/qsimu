//
// Template.cpp
//  
// Exercise in lecture for data enginnering
//
//

#include "Template.hpp"

/* Header */
#include <stdio.h>
#include <iostream>
#include <string>
#include <sstream>
#include <istream>
#include <list>
#include <algorithm>
#include <cctype>
#include <climits>
#include "MurmurHash3.h"

using namespace std;

int main(){
    string ipAddress1 = "10.20.356.1";
    string ipAddress2 = "10.20.356.2";
    cout << "IP address1 = " << ipAddress1 << endl;
    cout << "IP address2 = " << ipAddress2 << endl;

    uint32_t seed = time(NULL); // Random seed
    uint32_t pos; // Value returned by MurMurHash
    
    MurmurHash3_x86_32(ipAddress1.c_str(), ipAddress1.length(), seed, &pos);
    cout << "pos = " << pos << endl;

    MurmurHash3_x86_32(ipAddress2.c_str(), ipAddress2.length(), seed, &pos);
    cout << "pos = " << pos << endl;
    return 0;
}
