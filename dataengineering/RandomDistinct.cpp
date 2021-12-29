//
// RandomDistinct.cpp
//  
// Exercise in lecwure for data enginnering
//
//

#include "RandomDistinct.hpp"

/* Header */
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <istream>
#include <list>
#include <algorithm>
#include <cctype>
#include <climits>
#include <arpa/inet.h>
#include <regex>
#include "MurmurHash3.h"

using namespace std;

bool validateIpAddress(const string &ipAddress)
{
    struct sockaddr_in sa;
    int result = inet_pton(AF_INET, ipAddress.c_str(), &(sa.sin_addr));
    return result != 0;
}

int main(){

    uint32_t seed = time(NULL); // Random seed
    uint32_t min = 10000000;
    uint32_t M = 10000000;
    uint32_t pos; // Value returned by MurMurHash

    ifstream ifs("/Users/toshifumi.anan/uzabase/git/tanan/qsimu/dataengineering/benchmark.txt");
    string line;

    if (ifs.fail()) {
        cerr << "Failed to open file." << endl;
        return -1;
    }
    while (getline(ifs, line)) {
        string data = line.c_str();
        string ip = regex_replace(data.substr(0, data.find_last_not_of(" ")), regex(" "), ".");
        if (!(validateIpAddress(ip))) {
            continue;
        }
        MurmurHash3_x86_32(line.c_str(), line.length(), seed, &pos);
        if (min > pos) {
            min = pos;
        }
        if (M < pos) {
            M = pos;
        }
    }
    cout << "d = M / min = " << M / min << endl;
    return 0;
}
