using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Printing;

namespace IAUN.ComputerVision.FinalProject;

#pragma warning disable CA1416 // Validate platform compatibility
public class ImageProcessor
{
    public static (string bestFile, List<double> distances) CompareImages(string mainPath, string folderPath, List<int> distancesList, List<int> anglesDeg)
    {
        string[] files = Directory.GetFiles(folderPath);


        List<double[]> feats = [ExtractFeatures(mainPath, distancesList, anglesDeg)];
        foreach (var f in files)
        {
            var fileName = Path.GetFileName(f);
            Console.WriteLine($"{new string('-', 20)} {fileName} {new string('-', 20)}");

            feats.Add(ExtractFeatures(f, distancesList, anglesDeg));
        }

        double[][] mat = [.. feats];
        double[][] matNorm = ZScoreNormalize(mat);

        List<double> dists = [];
        var mainVec = matNorm[0];
        for (int i = 1; i < matNorm.Length; i++)
        {
            dists.Add(ManhattanDistance(mainVec, matNorm[i]));
        }


        int idxMin = 0;
        double minVal = dists[0];
        for (int i = 1; i < dists.Count; i++)
        {
            if (dists[i] < minVal)
            {
                minVal = dists[i];
                idxMin = i;
            }
        }

        string bestFile = Path.GetFileName(files[idxMin]);
        return (bestFile, dists);
    }
    private static double[] ExtractFeatures(string imgPath, List<int> distances, List<int> anglesDeg)
    {
        GetRgb(imgPath, out int H, out int W, out int[,,] rgb);
        
        int[,,] cmy = ConvertRgbToCMY(H, W, rgb);

        List<double> feats = [];
        var cmyDisplay = new[] { "C", "M", "Y" };
        Dictionary<string, double[]> values = [];
        for (int ch = 0; ch < 3; ch++)
        {
            var glcms = ComputeGLCM(cmy, ch, distances, anglesDeg, levels: 256);

            foreach (var g in glcms)
            {
                feats.AddRange(GlcmFeatures(g));
            }

            int[,] lbp = ComputeLBP(cmy, ch);
            var hist = LbpHistogram(lbp, nBins: 256);
            
            var quantizedLevel = 8;
            var quantizedHistogram = QuantizeHistogram(hist, quantizedLevel);
            values.Add(cmyDisplay[ch], quantizedHistogram);

            feats.AddRange(hist);

            PrintHist($"=== LBP Histogram – Channel {cmyDisplay[ch]} ===", hist);

        }
        foreach (var item in values)
        {
            Console.WriteLine($"quantized histogram for channel {item.Key}: {item.Value.Select(x => x.ToString("F3")).Aggregate(static (x, c) => x + "," + c)}");
        }

        return [.. feats];
    }
    private static void PrintHist(string title, double[] hist)
    {
        Console.WriteLine(title);
        for (int i = 0; i < hist.Length; i++)
            Console.WriteLine($" bin[{i,3}] = {hist[i]:F4}");
        Console.WriteLine();
    }
    private static double[] QuantizeHistogram(double[] hist256, int nBins = 8)
    {
        if (hist256 == null || hist256.Length != 256)
            throw new ArgumentException("Histogram should have 256 length.");

        if (256 % nBins != 0)
            throw new ArgumentException("256 must be divisible by nbins.");

        int step = 256 / nBins;
        double[] hist8 = new double[nBins];


        for (int i = 0; i < nBins; i++)
        {
            double sum = 0;
            int start = i * step;
            int end = start + step;
            for (int j = start; j < end; j++)
                sum += hist256[j];
            hist8[i] = sum;
        }

        double total = hist8.Sum();
        if (total > 0)
            for (int i = 0; i < nBins; i++)
                hist8[i] /= total;

        return hist8;
    }
    private static int[,,] ConvertRgbToCMY(int H, int W, int[,,] rgb)
    {
        int[,,] cmy = new int[H, W, 3];
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                for (int ch = 0; ch < 3; ch++)
                    cmy[y, x, ch] = 255 - rgb[y, x, ch];
        return cmy;
    }

    private static void GetRgb(string imgPath, out int H, out int W, out int[,,] rgb)
    {
        Bitmap bmp = new(imgPath);
        H = bmp.Height;
        W = bmp.Width;
        rgb = new int[H, W, 3];
        for (int y = 0; y < H; y++)
        {
            for (int x = 0; x < W; x++)
            {
                Color c = bmp.GetPixel(x, y);
                rgb[y, x, 0] = c.R;
                rgb[y, x, 1] = c.G;
                rgb[y, x, 2] = c.B;
            }
        }
    }

    private static List<double[,]> ComputeGLCM(int[,,] cmy, int channel, List<int> distances, List<int> anglesDeg, int levels = 256)
    {
        var H = cmy.GetLength(0);
        var W = cmy.GetLength(1);
        var result = new List<double[,]>();

        foreach (int d in distances)
        {
            foreach (int theta in anglesDeg)
            {

                double rad = theta * Math.PI / 180.0;
                int dx = (int)Math.Round(d * Math.Cos(rad));
                int dy = (int)Math.Round(-d * Math.Sin(rad));

                double[,] glcm = new double[levels, levels];

                int rStart = Math.Max(0, -dy);
                int rEnd = Math.Min(H, H - dy);
                int cStart = Math.Max(0, -dx);
                int cEnd = Math.Min(W, W - dx);

                double total = 0.0;
                for (int r = rStart; r < rEnd; r++)
                {
                    for (int c = cStart; c < cEnd; c++)
                    {
                        int iVal = cmy[r, c, channel];
                        int jVal = cmy[r + dy, c + dx, channel];
                        glcm[iVal, jVal] += 1.0;
                        total += 1.0;
                    }
                }
                if (total > 0)
                {
                    for (int i = 0; i < levels; i++)
                    {
                        for (int j = 0; j < levels; j++)
                        {
                            glcm[i, j] /= total;
                        }

                    }
                }
                result.Add(glcm);
            }
        }
        return result;
    }
    private static double[] GlcmFeatures(double[,] glcm)
    {
        var L = glcm.GetLength(0);
        double contrast = 0, dissimilarity = 0, homogeneity = 0, energy = 0, entropy = 0;
        double eps = 1e-12;

        for (int i = 0; i < L; i++)
        {
            for (int j = 0; j < L; j++)
            {
                double p = glcm[i, j];
                int diff = Math.Abs(i - j);
                contrast += diff * diff * p;
                dissimilarity += diff * p;
                homogeneity += p / (1.0 + diff);
                energy += p * p;
                if (p > eps)
                {
                    entropy -= p * Math.Log(p + eps, 2);

                }
            }
        }

        return [contrast, dissimilarity, homogeneity, energy, entropy];
    }

    private static int[,] ComputeLBP(int[,,] cmy, int channel)
    {
        int H = cmy.GetLength(0), W = cmy.GetLength(1);
        int[,] lbp = new int[H, W];

        int[,] offsets = new int[,] { { 0, 1 }, { -1, 1 }, { -1, 0 }, { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 } };

        for (int r = 0; r < H; r++)
        {
            for (int c = 0; c < W; c++)
            {
                int center = cmy[r, c, channel];
                int code = 0;
                for (int k = 0; k < 8; k++)
                {
                    int dr = offsets[k, 0], dc = offsets[k, 1];
                    int nr = r + dr, nc = c + dc;
                    int neigh = 0;
                    if (nr >= 0 && nr < H && nc >= 0 && nc < W)
                        neigh = cmy[nr, nc, channel];
                    if (neigh >= center)
                        code |= (1 << k);
                }
                lbp[r, c] = code;
            }
        }
        return lbp;
    }
    private static double[] LbpHistogram(int[,] lbp, int nBins = 256)
    {
        int H = lbp.GetLength(0);
        int W = lbp.GetLength(1);

        double[] hist = new double[nBins];

        int step = 256 / nBins;

        for (int r = 0; r < H; r++)
        {
            for (int c = 0; c < W; c++)
            {
                int code = lbp[r, c];
                int bin = code / step;
                if (bin >= nBins) bin = nBins - 1;
                hist[bin] += 1.0;
            }
        }
        double sum = hist.Sum();
        if (sum > 0)
        {
            for (int i = 0; i < nBins; i++)
                hist[i] /= sum;
        }

        return hist;
    }
    private static double[][] ZScoreNormalize(double[][] mat)
    {
        var N = mat.Length;
        var D = mat[0].Length;
        var mean = new double[D];
        var std = new double[D];

        for (int j = 0; j < D; j++)
        {
            for (int i = 0; i < N; i++)
            {
                mean[j] += mat[i][j];
            }
        }

        for (int j = 0; j < D; j++)
        {
            mean[j] /= N;
        }

        for (int j = 0; j < D; j++)
        {
            for (int i = 0; i < N; i++)
            {
                double diff = mat[i][j] - mean[j];
                std[j] += diff * diff;
            }
        }

        for (int j = 0; j < D; j++)
        {
            std[j] = Math.Sqrt(std[j] / N);
            if (std[j] < 1e-12) std[j] = 1.0;
        }

        var outMat = new double[N][];
        for (int i = 0; i < N; i++)
        {
            outMat[i] = new double[D];
            for (int j = 0; j < D; j++)
                outMat[i][j] = (mat[i][j] - mean[j]) / std[j];
        }
        return outMat;
    }
    static double ManhattanDistance(double[] a, double[] b)
    {
        double sum = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += Math.Abs(a[i] - b[i]);
        }
        return sum;
    }
}
