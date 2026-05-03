package lab04;

import net.sourceforge.jFuzzyLogic.FIS;
import net.sourceforge.jFuzzyLogic.rule.Variable;

import java.io.FileWriter;
import java.io.IOException;

public class InvestmentEvaluator {
    public static void main(String[] args) {
        String fcl = "investment.fcl";
        FIS fis = FIS.load(fcl, true);

        double[][] cases = new double[][]{
                {3.0, 10.0, 2.0},   // low return, high risk, low liquidity
                {3.0, 1.0, 2.0},    // low return, low risk, low liquidity
                {10.0, 5.0, 5.0},   // medium return, medium risk, medium liquidity
                {20.0, 1.0, 8.0},   // high return, low risk, high liquidity
                {20.0, 5.0, 5.0},   // high return, medium risk, medium liquidity
                {10.0, 1.0, 5.0},   // medium return, low risk, medium liquidity
                {20.0, 10.0, 2.0},  // high return, high risk, low liquidity
                {10.0, 10.0, 2.0},  // medium return, high risk, low liquidity
                {3.0, 1.0, 8.0},    // low return, low risk, high liquidity
                {10.0, 5.0, 8.0},   // medium return, medium risk, high liquidity
                {20.0, 10.0, 5.0}   // high return, high risk, medium liquidity
        };

        try (FileWriter csv = new FileWriter("results.csv")) {
            csv.write("case,expected_return,risk,liquidity,attractiveness\n");
            for (int i = 0; i < cases.length; i++) {
                double er = cases[i][0];
                double r = cases[i][1];
                double l = cases[i][2];

                fis.setVariable("expected_return", er);
                fis.setVariable("risk", r);
                fis.setVariable("liquidity", l);
                fis.evaluate();

                Variable out = fis.getVariable("attractiveness");
                double val = out.getValue();

                String line = String.format(java.util.Locale.US,
                        "%d,%.3f,%.3f,%.3f,%.3f\n",
                        i + 1, er, r, l, val);
                csv.write(line);
                System.out.printf("Case %d: ER=%.1f R=%.1f L=%.1f -> attractiveness=%.3f\n",
                        i + 1, er, r, l, val);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
