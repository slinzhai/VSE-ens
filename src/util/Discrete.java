package util;

/**
 * @author Songlin Zhai
 * 27 May 2018
 */

public class Discrete {
    private double[] Y;
    private int[] A;
    private int N;

    public int random() {
        // i = random uniform integer from { 1, 2, ..., N }
        int i = 1 + (int) (N * Math.random());
        double r = Math.random();
        if (r > Y[i]) i = A[i];
        return i;
    }

    public Discrete(double[] X) {
        N = X.length - 2;
        Y = new double[N+2];
        for (int i = 1; i <= N; i++) Y[i] = X[i];
        A = new int[N+2];
        int[] B = new int[N+2];   // for bookkeeping
        for(int i = 1; i <= N; i++) { 
            A[i] = B[i] = i;            // initial destins = stay there
            Y[i] = Y[i] * N;            // scale probability vector
        }

        // sentinels
        B[0]   = 0;
        B[N+1] = N+1;
        Y[0]   = 0.0;
        Y[N+1] = 2.0;

        int i = 0; 
        int j = N + 1;
        while(true) {
            do{ i++; } while(Y[B[i]] <  1.0);  // find i so X[B[i]] needs more
            do{ j--; } while(Y[B[j]] >= 1.0);  // find j so X[B[j]] wants less
            if(i >= j) break;
            int k = B[i]; B[i] = B[j]; B[j] = k;   // swap B[i] and B[j]
        }
        i = j;
        j++;
        while(i > 0) {
            while(Y[B[j]] <= 1.0) { j++; }  // find j so X[B[j]] needs more
            if(j > N) break;
            Y[B[j]] -= 1.0 - Y[B[i]];       // B[i] will donate to B[j] to fix up
            A[B[i]] = B[j];             
            if(Y[B[j]] < 1.0) {             // X[B[j]] now wants less so readjust ordering
                int k = B[i]; B[i] = B[j]; B[j] = k;   // swap B[j] and B[i]
                j++;
            }
            else { i--; }
        }
    }
}
