#include <bits/stdc++.h>
#include <cstdlib>
#include <stdlib.h>
using namespace std;
#define ull unsigned long long 
#define ld long double
#define ll long long
#define int long long
#define pii pair<int, int>
#define all(v) v.begin(), v.end()
int mod = 1e9 + 7; 
const ll inf = 1e18;
const int N = 1e6 + 50, K = 22;

using cd = complex<double>;
const double PI = acos(-1);

void fft(vector<cd> &a, bool invert){
    int n = a.size();

    for(int i = 1, j = 0; i < n; i++){
        int bit = n >> 1;
        for(; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if(i < j)
            swap(a[i], a[j]);
    }

    for(int len = 2; len <= n; len <<= 1){
        double ang = 2 * PI / len * (invert ? -1 : 1);
        cd wlen(cos(ang), sin(ang));

        for(int i = 0; i < n; i += len){
            cd w(1);
            for(int j = 0; j < len / 2; j++){
                cd u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if(invert){
        for(cd & x : a)
            x /= n;
    }
}

vector<ll> multiply_polynomials(const vector<int>& a, const vector<int>& b){
    vector<cd> fa(all(a)), fb(all(b));
    int n = 1;
    while(n < a.size() + b.size())
        n <<= 1;

    fa.resize(n);
    fb.resize(n);

    fft(fa, false);
    fft(fb, false);

    for(int i = 0; i < n; i++)
        fa[i] *= fb[i];

    fft(fa, true);

    vector<ll> res(n);
    for(int i = 0; i < n; i++)
        res[i] = round(fa[i].real());

    return res;
}

void solve(){  
    vector<cd> signal = {1, 2, 3, 4, 0, 0, 0, 0};
    cout << "Original Signal: ";
    for(auto val : signal) 
        cout << val.real() << ' ';
    cout << '\n';

    // Forward FFT
    fft(signal, false);
    cout << "After Forward FFT (Frequency Domain):\n";
    for(auto val : signal)
        cout << fixed << setprecision(2) << "(" << val.real() << ", " << val.imag() << "i) ";
    cout << '\n';

    fft(signal, true);
    cout << "After Inverse FFT (Reconstructed Signal): ";
    for(auto val : signal)
        cout << round(val.real()) << " ";
    cout << "\n\n";

    vector<int> p1 = {1, 2, 3};
    vector<int> p2 = {2, 4};

    vector<long long> prod = multiply_polynomials(p1, p2);
    cout << "(1 + 2x + 3x^2) * (2 + 4x) = \n";
    for(size_t i = 0; i < prod.size(); i++)
        if(prod[i] != 0){
            cout << prod[i] << "x^" << i << " ";
            
            if(i < prod.size() - 1) 
                cout << "+ ";
        }
    cout << '\n';
}


int32_t main(){
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    int t = 1;
    // cin >> t;
    while(t--)
        solve();
        
    return 0;
}