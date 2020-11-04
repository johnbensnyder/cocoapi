#include <mpi.h>
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <chrono>

PYBIND11_MAKE_OPAQUE(std::vector<uint>);

namespace py = pybind11;

struct image_struct {
  size_t id;
  int h;
  int w;
};

struct anns_struct {
  int image_id;
  int category_id;
  size_t id;
  float area;
  int iscrowd;
  float score;
  std::vector<float> bbox;
  // segmentation
  std::vector<std::vector<double>> segm_list;
  std::vector<int> segm_size;
  std::vector<uint> segm_counts_list;
  std::string segm_counts_str;
};

// create index results
std::vector<int64_t> imgids;
std::vector<int64_t> catids;
std::unordered_map<size_t, image_struct> imgsgt;
// std::unordered_map<size_t, image_struct> imgsdt;
// std::unordered_map<size_t, anns_struct> annsgt;
// std::unordered_map<size_t, anns_struct> annsdt;
// std::unordered_map<size_t, std::vector<anns_struct>> gtimgToAnns;
// std::unordered_map<size_t, std::vector<anns_struct>> dtimgToAnns;

// dict type
struct data_struct {
  std::vector<float> area;
  std::vector<int> iscrowd;
  std::vector<std::vector<float>> bbox;
  std::vector<int> ignore;
  std::vector<float> score;
  std::vector<std::vector<int>> segm_size;
  std::vector<std::vector<uint>> segm_counts;  // Change to RLE
  std::vector<int64_t> id;
};

// internal prepare results
inline size_t key(int i, int j) { return (size_t)i << 32 | (unsigned int)j; }
std::unordered_map<size_t, data_struct> gts_map;
std::unordered_map<size_t, data_struct> dts_map;
std::unordered_map<size_t, size_t> gtinds;
std::unordered_map<size_t, size_t> dtinds;

// internal computeiou results
std::unordered_map<size_t, std::vector<double>> ious_map;

template <typename T>
std::vector<size_t> sort_indices(std::vector<T> &v) {
  std::vector<size_t> indices(v.size());
  std::iota(indices.begin(), indices.end(), 0);

  // Note > instead of < in comparator for descending sort
  std::sort(indices.begin(), indices.end(),
            [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

  return indices;
}

void accumulate(int T, int A, std::vector<int> &maxDets,
                std::vector<double> &recThrs, std::vector<double> &precision,
                std::vector<double> &recall, std::vector<double> &scores, int K,
                int I, int R, int M, int k, int a,
                std::vector<std::vector<int64_t>> &gtignore,
                std::vector<std::vector<double>> &dtignore,
                std::vector<std::vector<double>> &dtmatches,
                std::vector<std::vector<double>> &dtscores);

void accumulate_dist(int T, int A, std::vector<int> &maxDets,
                     std::vector<double> &recThrs,
                     std::vector<double> &precision,
                     std::vector<double> &recall, std::vector<double> &scores,
                     int K, int I, int R, int M, int k, int a,
                     std::vector<std::vector<int64_t>> &gtignore,
                     std::vector<std::vector<double>> &dtignore,
                     std::vector<std::vector<double>> &dtmatches,
                     std::vector<std::vector<double>> &dtscores,
                     MPI_Comm accumuate_comm);

void compute_iou(std::string iouType, int maxDet, int useCats);

std::tuple<py::array_t<int64_t>, py::array_t<int64_t>, py::dict> cpp_evaluate(
    int useCats, std::vector<std::vector<double>> areaRngs,
    std::vector<double> iouThrs_ptr, std::vector<int> maxDets,
    std::vector<double> recThrs, std::string iouType, int nthreads) {
  assert(useCats > 0);

  int T = iouThrs_ptr.size();
  int A = areaRngs.size();
  int K = catids.size();
  int R = recThrs.size();
  int M = maxDets.size();
  std::vector<double> precision(T * R * K * A * M);
  std::vector<double> recall(T * K * A * M);
  std::vector<double> scores(T * R * K * A * M);

  int maxDet = maxDets[M - 1];
  compute_iou(iouType, maxDet, useCats);

#pragma omp parallel for num_threads(nthreads)
  for (size_t c = 0; c < catids.size(); c++) {
    for (size_t a = 0; a < areaRngs.size(); a++) {
      std::vector<std::vector<int64_t>> gtIgnore_list;
      std::vector<std::vector<double>> dtIgnore_list;
      std::vector<std::vector<double>> dtMatches_list;
      std::vector<std::vector<double>> dtScores_list;
      for (size_t i = 0; i < imgids.size(); i++) {
        int catId = catids[c];
        int imgId = imgids[i];

        auto gtsm = &gts_map[key(imgId, catId)];
        auto dtsm = &dts_map[key(imgId, catId)];

        if ((gtsm->id.size() == 0) && (dtsm->id.size() == 0)) {
          continue;
        }

        double aRng0 = areaRngs[a][0];
        double aRng1 = areaRngs[a][1];
        // sizes
        int T = iouThrs_ptr.size();
        int G = gtsm->id.size();
        int Do = dtsm->id.size();
        int D = std::min(Do, maxDet);
        int I = (G == 0 || D == 0) ? 0 : D;
        // arrays
        std::vector<size_t> gtind(G);
        std::vector<size_t> dtind(Do);
        std::vector<int> iscrowd(G);
        std::vector<double> gtm(T * G);
        std::vector<int64_t> gtIds(G);
        std::vector<int64_t> dtIds(D);
        gtIgnore_list.push_back(std::vector<int64_t>(G));
        dtIgnore_list.push_back(std::vector<double>(T * D));
        dtMatches_list.push_back(std::vector<double>(T * D));
        dtScores_list.push_back(std::vector<double>(D));
        // pointers
        auto gtIg = &gtIgnore_list.back()[0];
        auto dtIg = &dtIgnore_list.back()[0];
        auto dtm = &dtMatches_list.back()[0];
        auto dtScores = &dtScores_list.back()[0];
        auto ious = &ious_map[key(imgId, catId)][0];
        // set ignores
        for (int g = 0; g < G; g++) {
          double area = gtsm->area[g];
          int64_t ignore = gtsm->ignore[g];
          if (ignore || (area < aRng0 || area > aRng1)) {
            gtIg[g] = 1;
          } else {
            gtIg[g] = 0;
          }
        }
        // sort dt highest score first, sort gt ignore last
        std::vector<float> ignores;
        for (int g = 0; g < G; g++) {
          auto ignore = gtIg[g];
          ignores.push_back(ignore);
        }
        auto g_indices = sort_indices(ignores);
        for (int g = 0; g < G; g++) {
          gtind[g] = g_indices[g];
        }
        std::vector<float> scores;
        for (int d = 0; d < Do; d++) {
          auto score = -dtsm->score[d];
          scores.push_back(score);
        }
        auto indices = sort_indices(scores);
        for (int d = 0; d < Do; d++) {
          dtind[d] = indices[d];
        }
        // iscrowd and ignores
        for (int g = 0; g < G; g++) {
          iscrowd[g] = gtsm->iscrowd[gtind[g]];
          gtIg[g] = ignores[gtind[g]];
        }
        // zero arrays
        for (int t = 0; t < T; t++) {
          for (int g = 0; g < G; g++) {
            gtm[t * G + g] = 0;
          }
          for (int d = 0; d < D; d++) {
            dtm[t * D + d] = 0;
            dtIg[t * D + d] = 0;
          }
        }
        // if not len(ious)==0:
        if (I != 0) {
          for (int t = 0; t < T; t++) {
            double thresh = iouThrs_ptr[t];
            for (int d = 0; d < D; d++) {
              double iou = thresh < (1 - 1e-10) ? thresh : (1 - 1e-10);
              int m = -1;
              for (int g = 0; g < G; g++) {
                // if this gt already matched, and not a crowd, continue
                if ((gtm[t * G + g] > 0) && (iscrowd[g] == 0)) continue;
                // if dt matched to reg gt, and on ignore gt, stop
                if ((m > -1) && (gtIg[m] == 0) && (gtIg[g] == 1)) break;
                // continue to next gt unless better match made
                double val = ious[d + I * gtind[g]];
                if (val < iou) continue;
                // if match successful and best so far, store appropriately
                iou = val;
                m = g;
              }
              // if match made store id of match for both dt and gt
              if (m == -1) continue;
              dtIg[t * D + d] = gtIg[m];
              dtm[t * D + d] = gtsm->id[gtind[m]];
              gtm[t * G + m] = dtsm->id[dtind[d]];
            }
          }
        }
        // set unmatched detections outside of area range to ignore
        for (int d = 0; d < D; d++) {
          float val = dtsm->area[dtind[d]];
          double x3 = (val < aRng0 ||
                       val > aRng1);  // why does this need to be a double
          for (int t = 0; t < T; t++) {
            double x1 = dtIg[t * D + d];
            double x2 = dtm[t * D + d];
            double res = x1 || ((x2 == 0) && x3);
            dtIg[t * D + d] = res;
          }
        }
        // store results for given image and category
        for (int g = 0; g < G; g++) {
          gtIds[g] = gtsm->id[gtind[g]];
        }
        for (int d = 0; d < D; d++) {
          dtIds[d] = dtsm->id[dtind[d]];
          dtScores[d] = dtsm->score[dtind[d]];
        }
      }
      // accumulate
      accumulate(iouThrs_ptr.size(), areaRngs.size(), maxDets, recThrs,
                 precision, recall, scores, catids.size(), imgids.size(),
                 recThrs.size(), maxDets.size(), c, a, gtIgnore_list,
                 dtIgnore_list, dtMatches_list, dtScores_list);
    }
  }

  // clear arrays
  std::unordered_map<size_t, std::vector<double>>().swap(ious_map);
  // std::unordered_map<size_t, data_struct>().swap(gts_map);
  std::unordered_map<size_t, data_struct>().swap(dts_map);
  // std::unordered_map<size_t, image_struct>().swap(imgsdt);
  // std::unordered_map<size_t, anns_struct>().swap(annsdt);
  // std::unordered_map<size_t, std::vector<anns_struct>>().swap(dtimgToAnns);

  // dictionary
  py::dict dictret;
  py::list l;
  l.append(T);
  l.append(R);
  l.append(K);
  l.append(A);
  l.append(M);
  dictret["counts"] = l;
  dictret["precision"] = py::array_t<double>(
      {T, R, K, A, M}, {R * K * A * M * 8, K * A * M * 8, A * M * 8, M * 8, 8},
      &precision[0]);
  dictret["recall"] = py::array_t<double>(
      {T, K, A, M}, {K * A * M * 8, A * M * 8, M * 8, 8}, &recall[0]);
  dictret["scores"] = py::array_t<double>(
      {T, R, K, A, M}, {R * K * A * M * 8, K * A * M * 8, A * M * 8, M * 8, 8},
      &scores[0]);

  py::array_t<int64_t> imgidsret =
      py::array_t<int64_t>({imgids.size()}, {8}, &imgids[0]);
  py::array_t<int64_t> catidsret =
      py::array_t<int64_t>({catids.size()}, {8}, &catids[0]);

  return std::tuple<py::array_t<int64_t>, py::array_t<int64_t>, py::dict>(
      imgidsret, catidsret, dictret);
}

std::tuple<py::array_t<int64_t>, py::array_t<int64_t>, py::dict>
cpp_evaluate_dist(int useCats, std::vector<std::vector<double>> areaRngs,
                  std::vector<double> iouThrs_ptr, std::vector<int> maxDets,
                  std::vector<double> recThrs, std::string iouType,
                  int nthreads, std::vector<int64_t> imgids, int dist) {
  assert(useCats > 0);

  MPI_Comm accumulate_comm;
  if (dist >= 2) {
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the group of processes in MPI_COMM_WORLD
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    int group_size = int(world_size / 8);
    int ranks[group_size];
    for (auto i = 0; i < group_size; ++i) {
      ranks[i] = (i * 8) + (dist - 2);
    }
    // Construct a group containing all of the prime ranks in world_group
    MPI_Group hier_group;
    MPI_Group_incl(world_group, group_size, ranks, &hier_group);
    // Create a new communicator based on the group
    MPI_Comm_create_group(MPI_COMM_WORLD, hier_group, 0, &accumulate_comm);
  } else {
    accumulate_comm = MPI_COMM_WORLD;
  }

  // nthreads = 1;
  // MPI_Comm accumulate_comm[nthreads];
  // MPI_Request reqs[nthreads];
  // MPI_Status array_of_statuses[nthreads];
  // for (int i = 0; i < nthreads; ++i) {
  //   MPI_Comm_idup(MPI_COMM_WORLD, &accumulate_comm[i], &reqs[i]);
  // }
  // int ret = MPI_Waitall(nthreads, reqs, array_of_statuses);
  // assert(ret == MPI_SUCCESS);

  int T = iouThrs_ptr.size();
  int A = areaRngs.size();
  int K = catids.size();
  int R = recThrs.size();
  int M = maxDets.size();
  std::vector<double> precision(T * R * K * A * M);
  std::vector<double> recall(T * K * A * M);
  std::vector<double> scores(T * R * K * A * M);

  int maxDet = maxDets[M - 1];
  compute_iou(iouType, maxDet, useCats);

  //#pragma omp parallel for num_threads(nthreads)
  for (size_t c = 0; c < catids.size(); c++) {
    for (size_t a = 0; a < areaRngs.size(); a++) {
      std::vector<std::vector<int64_t>> gtIgnore_list;
      std::vector<std::vector<double>> dtIgnore_list;
      std::vector<std::vector<double>> dtMatches_list;
      std::vector<std::vector<double>> dtScores_list;
      // std::cout << "Imgids size is " << imgids.size() << std::endl;
      for (size_t i = 0; i < imgids.size(); i++) {
        int catId = catids[c];
        int imgId = imgids[i];

        auto gtsm = &gts_map[key(imgId, catId)];
        auto dtsm = &dts_map[key(imgId, catId)];

        if ((gtsm->id.size() == 0) && (dtsm->id.size() == 0)) {
          continue;
        }

        double aRng0 = areaRngs[a][0];
        double aRng1 = areaRngs[a][1];
        // sizes
        int T = iouThrs_ptr.size();
        int G = gtsm->id.size();
        int Do = dtsm->id.size();
        int D = std::min(Do, maxDet);
        int I = (G == 0 || D == 0) ? 0 : D;
        // arrays
        std::vector<size_t> gtind(G);
        std::vector<size_t> dtind(Do);
        std::vector<int> iscrowd(G);
        std::vector<double> gtm(T * G);
        std::vector<int64_t> gtIds(G);
        std::vector<int64_t> dtIds(D);
        gtIgnore_list.push_back(std::vector<int64_t>(G));
        dtIgnore_list.push_back(std::vector<double>(T * D));
        dtMatches_list.push_back(std::vector<double>(T * D));
        dtScores_list.push_back(std::vector<double>(D));
        // pointers
        auto gtIg = &gtIgnore_list.back()[0];
        auto dtIg = &dtIgnore_list.back()[0];
        auto dtm = &dtMatches_list.back()[0];
        auto dtScores = &dtScores_list.back()[0];
        auto ious = &ious_map[key(imgId, catId)][0];
        // set ignores
        for (int g = 0; g < G; g++) {
          double area = gtsm->area[g];
          int64_t ignore = gtsm->ignore[g];
          if (ignore || (area < aRng0 || area > aRng1)) {
            gtIg[g] = 1;
          } else {
            gtIg[g] = 0;
          }
        }
        // sort dt highest score first, sort gt ignore last
        std::vector<float> ignores;
        for (int g = 0; g < G; g++) {
          auto ignore = gtIg[g];
          ignores.push_back(ignore);
        }
        auto g_indices = sort_indices(ignores);
        for (int g = 0; g < G; g++) {
          gtind[g] = g_indices[g];
        }
        std::vector<float> scores;
        for (int d = 0; d < Do; d++) {
          auto score = -dtsm->score[d];
          scores.push_back(score);
        }
        auto indices = sort_indices(scores);
        for (int d = 0; d < Do; d++) {
          dtind[d] = indices[d];
        }
        // iscrowd and ignores
        for (int g = 0; g < G; g++) {
          iscrowd[g] = gtsm->iscrowd[gtind[g]];
          gtIg[g] = ignores[gtind[g]];
        }
        // zero arrays
        for (int t = 0; t < T; t++) {
          for (int g = 0; g < G; g++) {
            gtm[t * G + g] = 0;
          }
          for (int d = 0; d < D; d++) {
            dtm[t * D + d] = 0;
            dtIg[t * D + d] = 0;
          }
        }
        // if not len(ious)==0:
        if (I != 0) {
          for (int t = 0; t < T; t++) {
            double thresh = iouThrs_ptr[t];
            for (int d = 0; d < D; d++) {
              double iou = thresh < (1 - 1e-10) ? thresh : (1 - 1e-10);
              int m = -1;
              for (int g = 0; g < G; g++) {
                // if this gt already matched, and not a crowd, continue
                if ((gtm[t * G + g] > 0) && (iscrowd[g] == 0)) continue;
                // if dt matched to reg gt, and on ignore gt, stop
                if ((m > -1) && (gtIg[m] == 0) && (gtIg[g] == 1)) break;
                // continue to next gt unless better match made
                double val = ious[d + I * gtind[g]];
                if (val < iou) continue;
                // if match successful and best so far, store appropriately
                iou = val;
                m = g;
              }
              // if match made store id of match for both dt and gt
              if (m == -1) continue;
              dtIg[t * D + d] = gtIg[m];
              dtm[t * D + d] = gtsm->id[gtind[m]];
              gtm[t * G + m] = dtsm->id[dtind[d]];
            }
          }
        }
        // set unmatched detections outside of area range to ignore
        for (int d = 0; d < D; d++) {
          float val = dtsm->area[dtind[d]];
          double x3 = (val < aRng0 || val > aRng1);
          for (int t = 0; t < T; t++) {
            double x1 = dtIg[t * D + d];
            double x2 = dtm[t * D + d];
            double res = x1 || ((x2 == 0) && x3);
            dtIg[t * D + d] = res;
          }
        }
        // store results for given image and category
        for (int g = 0; g < G; g++) {
          gtIds[g] = gtsm->id[gtind[g]];
        }
        for (int d = 0; d < D; d++) {
          dtIds[d] = dtsm->id[dtind[d]];
          dtScores[d] = dtsm->score[dtind[d]];
        }
      }

      // accumulate
      accumulate_dist(iouThrs_ptr.size(), areaRngs.size(), maxDets, recThrs,
                      precision, recall, scores, catids.size(), imgids.size(),
                      recThrs.size(), maxDets.size(), c, a, gtIgnore_list,
                      dtIgnore_list, dtMatches_list, dtScores_list,
                      accumulate_comm);
    }
  }

  // clear arrays
  std::unordered_map<size_t, std::vector<double>>().swap(ious_map);
  // std::unordered_map<size_t, data_struct>().swap(gts_map);
  std::unordered_map<size_t, data_struct>().swap(dts_map);
  // std::unordered_map<size_t, image_struct>().swap(imgsdt);
  // std::unordered_map<size_t, anns_struct>().swap(annsdt);
  // std::unordered_map<size_t, std::vector<anns_struct>>().swap(dtimgToAnns);

  // dictionary
  py::dict dictret;
  py::list l;
  l.append(T);
  l.append(R);
  l.append(K);
  l.append(A);
  l.append(M);
  dictret["counts"] = l;
  dictret["precision"] = py::array_t<double>(
      {T, R, K, A, M}, {R * K * A * M * 8, K * A * M * 8, A * M * 8, M * 8, 8},
      &precision[0]);
  dictret["recall"] = py::array_t<double>(
      {T, K, A, M}, {K * A * M * 8, A * M * 8, M * 8, 8}, &recall[0]);
  dictret["scores"] = py::array_t<double>(
      {T, R, K, A, M}, {R * K * A * M * 8, K * A * M * 8, A * M * 8, M * 8, 8},
      &scores[0]);

  py::array_t<int64_t> imgidsret =
      py::array_t<int64_t>({imgids.size()}, {8}, &imgids[0]);
  py::array_t<int64_t> catidsret =
      py::array_t<int64_t>({catids.size()}, {8}, &catids[0]);

  return std::tuple<py::array_t<int64_t>, py::array_t<int64_t>, py::dict>(
      imgidsret, catidsret, dictret);
}

template <typename T>
std::vector<size_t> stable_sort_indices(std::vector<T> &v) {
  std::vector<size_t> indices(v.size());
  std::iota(indices.begin(), indices.end(), 0);

  // Note > instead of < in comparator for descending sort
  std::stable_sort(indices.begin(), indices.end(),
                   [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

  return indices;
}

template <typename T>
std::vector<T> assemble_array(std::vector<std::vector<T>> &list, size_t nrows,
                              size_t maxDet, std::vector<size_t> &indices) {
  std::vector<T> q;
  // Need to get N_rows from an entry in order to compute output size
  // copy first maxDet entries from each entry -> array
  for (size_t e = 0; e < list.size(); ++e) {
    auto arr = list[e];
    size_t cols = arr.size() / nrows;
    size_t ncols = std::min(maxDet, cols);
    for (size_t j = 0; j < ncols; ++j) {
      for (size_t i = 0; i < nrows; ++i) {
        q.push_back(arr[i * cols + j]);
      }
    }
  }
  // now we've done that, copy the relevant entries based on indices
  std::vector<T> res(indices.size() * nrows);
  for (size_t i = 0; i < nrows; ++i) {
    for (size_t j = 0; j < indices.size(); ++j) {
      res[i * indices.size() + j] = q[indices[j] * nrows + i];
    }
  }

  return res;
}

std::vector<double> GatherArrays(std::vector<double> q, int world_size,
                                 int world_rank, MPI_Comm accum_comm) {
  // Gather arrays before getting sorted
  // Gather sizes They can be different
  std::vector<int> array_sizes(world_size);
  size_t qsize = q.size();
  int mpi_ret = MPI_Gather(&qsize, 1, MPI_INT, array_sizes.data(), 1, MPI_INT,
                           0, accum_comm);
  assert(mpi_ret == MPI_SUCCESS);
  int total_size = 0;
  std::vector<int> displacements(world_size, 0);
  if (world_rank == 0) {
    for (int ii = 0; ii < world_size; ii++) {
      displacements[ii] = total_size;
      total_size += array_sizes[ii];
    }
  }

  std::vector<double> q_total(total_size, 0.0);

  // Gatherv
  mpi_ret = MPI_Gatherv(q.data(), q.size(), MPI_DOUBLE, q_total.data(),
                        array_sizes.data(), displacements.data(), MPI_DOUBLE, 0,
                        accum_comm);
  assert(mpi_ret == MPI_SUCCESS);

  return q_total;
}

std::vector<double> assemble_array_dist(std::vector<std::vector<double>> &list,
                                        size_t nrows, size_t maxDet,
                                        std::vector<size_t> &indices,
                                        int world_size, int world_rank,
                                        MPI_Comm accum_comm) {
  std::vector<double> q;
  // Need to get N_rows from an entry in order to compute output size
  // copy first maxDet entries from each entry -> array
  for (size_t e = 0; e < list.size(); ++e) {
    auto arr = list[e];
    size_t cols = arr.size() / nrows;
    size_t ncols = std::min(maxDet, cols);
    for (size_t j = 0; j < ncols; ++j) {
      for (size_t i = 0; i < nrows; ++i) {
        q.push_back(arr[i * cols + j]);
      }
    }
  }
  // std::vector<double> newq =
  q = GatherArrays(q, world_size, world_rank, accum_comm);

  // now we've done that, copy the relevant entries based on indices
  std::vector<double> res(indices.size() * nrows);
  for (size_t i = 0; i < nrows; ++i) {
    for (size_t j = 0; j < indices.size(); ++j) {
      res[i * indices.size() + j] = q[indices[j] * nrows + i];
    }
  }

  return res;
}

void accumulate_dist(int T, int A, std::vector<int> &maxDets,
                     std::vector<double> &recThrs,
                     std::vector<double> &precision,
                     std::vector<double> &recall, std::vector<double> &scores,
                     int K, int I, int R, int M, int k, int a,
                     std::vector<std::vector<int64_t>> &gtignore,
                     std::vector<std::vector<double>> &dtignore,
                     std::vector<std::vector<double>> &dtmatches,
                     std::vector<std::vector<double>> &dtscores,
                     MPI_Comm accumulate_comm) {
  // if (dtscores.size() == 0) return;
  if (accumulate_comm == MPI_COMM_NULL) return;
  int world_size;
  // MPI_Comm_dup(MPI_COMM_WORLD, &accumulate_comm);
  MPI_Comm_size(accumulate_comm, &world_size);
  int world_rank;
  MPI_Comm_rank(accumulate_comm, &world_rank);
  int mpi_ret;

  // Allreduce sum for dtscores size

  // int dt_size = dtscores.size();
  // int all_dt_size;
  // mpi_ret = MPI_Allreduce(&dt_size, &all_dt_size, 1, MPI_INT, MPI_SUM,
  //                         accumulate_comm);
  // assert(MPI_SUCCESS == mpi_ret);
  // if (all_dt_size == 0) return;

  for (int m = 0; m < M; ++m) {
    // gtIg = np.concatenate([e['gtIgnore'] for e in E])
    // npig = np.count_nonzero(gtIg==0 )
    int npig = 0;
    for (size_t e = 0; e < gtignore.size(); ++e) {
      auto ignore = gtignore[e];
      for (size_t j = 0; j < ignore.size(); ++j) {
        if (ignore[j] == 0) npig++;
      }
    }
    int npigather;
    mpi_ret =
        MPI_Allreduce(&npig, &npigather, 1, MPI_INT, MPI_SUM, accumulate_comm);
    assert(mpi_ret == MPI_SUCCESS);
    npig = npigather;
    if (npig == 0) continue;

    auto maxDet = maxDets[m];
    // Concatenate first maxDet scores in each evalImg entry, -ve and sort
    // w/indices
    std::vector<double> dtScores;
    for (size_t e = 0; e < dtscores.size(); ++e) {
      auto score = dtscores[e];
      for (size_t j = 0; j < std::min(score.size(), (size_t)maxDet); ++j) {
        dtScores.push_back(-score[j]);
      }
    }

    // Gather dtmatches, dtignore, dtScores
    // Gather scores before its sorted, Inside assemble array gather

    // std::vector<double> newdtScores =
    dtScores = GatherArrays(dtScores, world_size, world_rank, accumulate_comm);

    // get sorted indices of scores
    auto indices = stable_sort_indices(dtScores);
    std::vector<double> dtScoresSorted(dtScores.size());
    for (size_t j = 0; j < indices.size(); ++j) {
      dtScoresSorted[j] = -dtScores[indices[j]];
    }

    auto dtm = assemble_array_dist(dtmatches, T, maxDet, indices, world_size,
                                   world_rank, accumulate_comm);
    auto dtIg = assemble_array_dist(dtignore, T, maxDet, indices, world_size,
                                    world_rank, accumulate_comm);

    if (world_rank != 0) {
      continue;
    }

    int nrows = indices.size() ? dtm.size() / indices.size() : 0;
    std::vector<double> tp_sum(indices.size() * nrows);
    std::vector<double> fp_sum(indices.size() * nrows);
#pragma omp parallel for num_threads(8)
    for (int i = 0; i < nrows; ++i) {
      size_t tsum = 0, fsum = 0;
      for (size_t j = 0; j < indices.size(); ++j) {
        int index = i * indices.size() + j;
        tsum += (dtm[index]) && (!dtIg[index]);
        fsum += (!dtm[index]) && (!dtIg[index]);
        tp_sum[index] = tsum;
        fp_sum[index] = fsum;
      }
    }

    double eps =
        2.220446049250313e-16;  // std::numeric_limits<double>::epsilon();
#pragma omp parallel for num_threads(8)
    for (int t = 0; t < nrows; ++t) {
      // nd = len(tp)
      int nd = indices.size();
      std::vector<double> rc(indices.size());
      std::vector<double> pr(indices.size());
      for (size_t j = 0; j < indices.size(); ++j) {
        int index = t * indices.size() + j;
        // rc = tp / npig
        rc[j] = tp_sum[index] / npig;
        // pr = tp / (fp+tp+np.spacing(1))
        pr[j] = tp_sum[index] / (fp_sum[index] + tp_sum[index] + eps);
      }

      recall[t * K * A * M + k * A * M + a * M + m] =
          nd ? rc[indices.size() - 1] : 0;

      std::vector<double> q(R);
      std::vector<double> ss(R);

      for (int i = 0; i < R; i++) {
        q[i] = 0;
        ss[i] = 0;
      }

      for (int i = nd - 1; i > 0; --i) {
        if (pr[i] > pr[i - 1]) {
          pr[i - 1] = pr[i];
        }
      }

      std::vector<int> inds(recThrs.size());
      for (size_t i = 0; i < recThrs.size(); i++) {
        auto it = std::lower_bound(rc.begin(), rc.end(), recThrs[i]);
        inds[i] = it - rc.begin();
      }

      for (size_t ri = 0; ri < inds.size(); ri++) {
        size_t pi = inds[ri];
        if (pi >= pr.size()) continue;
        q[ri] = pr[pi];
        ss[ri] = dtScoresSorted[pi];
      }

      for (size_t i = 0; i < inds.size(); i++) {
        // precision[t,:,k,a,m] = np.array(q)
        size_t index =
            t * R * K * A * M + i * K * A * M + k * A * M + a * M + m;
        precision[index] = q[i];
        scores[index] = ss[i];
      }
    }
  }
}

void accumulate(int T, int A, std::vector<int> &maxDets,
                std::vector<double> &recThrs, std::vector<double> &precision,
                std::vector<double> &recall, std::vector<double> &scores, int K,
                int I, int R, int M, int k, int a,
                std::vector<std::vector<int64_t>> &gtignore,
                std::vector<std::vector<double>> &dtignore,
                std::vector<std::vector<double>> &dtmatches,
                std::vector<std::vector<double>> &dtscores) {
  if (dtscores.size() == 0) return;

  for (int m = 0; m < M; ++m) {
    auto maxDet = maxDets[m];
    // Concatenate first maxDet scores in each evalImg entry, -ve and sort
    // w/indices
    std::vector<double> dtScores;
    for (size_t e = 0; e < dtscores.size(); ++e) {
      auto score = dtscores[e];
      for (size_t j = 0; j < std::min(score.size(), (size_t)maxDet); ++j) {
        dtScores.push_back(-score[j]);
      }
    }

    // get sorted indices of scores
    auto indices = stable_sort_indices(dtScores);
    std::vector<double> dtScoresSorted(dtScores.size());
    for (size_t j = 0; j < indices.size(); ++j) {
      dtScoresSorted[j] = -dtScores[indices[j]];
    }

    auto dtm = assemble_array<double>(dtmatches, T, maxDet, indices);
    auto dtIg = assemble_array<double>(dtignore, T, maxDet, indices);

    // gtIg = np.concatenate([e['gtIgnore'] for e in E])
    // npig = np.count_nonzero(gtIg==0 )
    int npig = 0;
    for (size_t e = 0; e < gtignore.size(); ++e) {
      auto ignore = gtignore[e];
      for (size_t j = 0; j < ignore.size(); ++j) {
        if (ignore[j] == 0) npig++;
      }
    }

    if (npig == 0) continue;

    int nrows = indices.size() ? dtm.size() / indices.size() : 0;
    std::vector<double> tp_sum(indices.size() * nrows);
    std::vector<double> fp_sum(indices.size() * nrows);
    for (int i = 0; i < nrows; ++i) {
      size_t tsum = 0, fsum = 0;
      for (size_t j = 0; j < indices.size(); ++j) {
        int index = i * indices.size() + j;
        tsum += (dtm[index]) && (!dtIg[index]);
        fsum += (!dtm[index]) && (!dtIg[index]);
        tp_sum[index] = tsum;
        fp_sum[index] = fsum;
      }
    }

    double eps =
        2.220446049250313e-16;  // std::numeric_limits<double>::epsilon();
    for (int t = 0; t < nrows; ++t) {
      // nd = len(tp)
      int nd = indices.size();
      std::vector<double> rc(indices.size());
      std::vector<double> pr(indices.size());
      for (size_t j = 0; j < indices.size(); ++j) {
        int index = t * indices.size() + j;
        // rc = tp / npig
        rc[j] = tp_sum[index] / npig;
        // pr = tp / (fp+tp+np.spacing(1))
        pr[j] = tp_sum[index] / (fp_sum[index] + tp_sum[index] + eps);
      }

      recall[t * K * A * M + k * A * M + a * M + m] =
          nd ? rc[indices.size() - 1] : 0;

      std::vector<double> q(R);
      std::vector<double> ss(R);

      for (int i = 0; i < R; i++) {
        q[i] = 0;
        ss[i] = 0;
      }

      for (int i = nd - 1; i > 0; --i) {
        if (pr[i] > pr[i - 1]) {
          pr[i - 1] = pr[i];
        }
      }

      std::vector<int> inds(recThrs.size());
      for (size_t i = 0; i < recThrs.size(); i++) {
        auto it = std::lower_bound(rc.begin(), rc.end(), recThrs[i]);
        inds[i] = it - rc.begin();
      }

      for (size_t ri = 0; ri < inds.size(); ri++) {
        size_t pi = inds[ri];
        if (pi >= pr.size()) continue;
        q[ri] = pr[pi];
        ss[ri] = dtScoresSorted[pi];
      }

      for (size_t i = 0; i < inds.size(); i++) {
        // precision[t,:,k,a,m] = np.array(q)
        size_t index =
            t * R * K * A * M + i * K * A * M + k * A * M + a * M + m;
        precision[index] = q[i];
        scores[index] = ss[i];
      }
    }
  }
}

void bbIou(double *dt, double *gt, int m, int n, int *iscrowd, double *o) {
  double h, w, i, u, ga, da;
  int g, d;
  int crowd;
  for (g = 0; g < n; g++) {
    double *G = gt + g * 4;
    ga = G[2] * G[3];
    crowd = iscrowd != NULL && iscrowd[g];
    for (d = 0; d < m; d++) {
      double *D = dt + d * 4;
      da = D[2] * D[3];
      o[g * m + d] = 0;
      w = fmin(D[2] + D[0], G[2] + G[0]) - fmax(D[0], G[0]);
      if (w <= 0) continue;
      h = fmin(D[3] + D[1], G[3] + G[1]) - fmax(D[1], G[1]);
      if (h <= 0) continue;
      i = w * h;
      u = crowd ? da : da + ga - i;
      o[g * m + d] = i / u;
    }
  }
}

typedef struct {
  unsigned long h, w, m;
  unsigned int *cnts;
} RLE;

void rleInit(RLE *R, unsigned long h, unsigned long w, unsigned long m,
             unsigned int *cnts) {
  R->h = h;
  R->w = w;
  R->m = m;
  R->cnts = (m == 0) ? 0 : (unsigned int *)malloc(sizeof(unsigned int) * m);
  unsigned long j;
  if (cnts)
    for (j = 0; j < m; j++) R->cnts[j] = cnts[j];
}

void rleFree(RLE *R) {
  free(R->cnts);
  R->cnts = 0;
}

void rleFrString(RLE *R, char *s, unsigned long h, unsigned long w) {
  unsigned long m = 0, p = 0, k;
  long x;
  int more;
  unsigned int *cnts;
  while (s[m]) m++;
  cnts = (unsigned int *)malloc(sizeof(unsigned int) * m);
  m = 0;
  while (s[p]) {
    x = 0;
    k = 0;
    more = 1;
    while (more) {
      char c = s[p] - 48;
      x |= (c & 0x1f) << 5 * k;
      more = c & 0x20;
      p++;
      k++;
      if (!more && (c & 0x10)) x |= -1 << 5 * k;
    }
    if (m > 2) x += (long)cnts[m - 2];
    cnts[m++] = (unsigned int)x;
  }
  rleInit(R, h, w, m, cnts);
  free(cnts);
}

void cntsFrString(std::vector<uint> &cnts, char *s, unsigned long h,
                  unsigned long w) {
  unsigned long m = 0, p = 0, k;
  long x;
  int more;
  m = 0;
  while (s[p]) {
    x = 0;
    k = 0;
    more = 1;
    while (more) {
      char c = s[p] - 48;
      x |= (c & 0x1f) << 5 * k;
      more = c & 0x20;
      p++;
      k++;
      if (!more && (c & 0x10)) x |= -1 << 5 * k;
    }
    if (m > 2) x += (long)cnts[m - 2];
    m++;
    cnts.push_back((unsigned int)x);
  }
}

unsigned int umin(unsigned int a, unsigned int b) { return (a < b) ? a : b; }
unsigned int umax(unsigned int a, unsigned int b) { return (a > b) ? a : b; }

void rleArea(const RLE *R, unsigned long n, unsigned int *a) {
  unsigned long i, j;
  for (i = 0; i < n; i++) {
    a[i] = 0;
    for (j = 1; j < R[i].m; j += 2) a[i] += R[i].cnts[j];
  }
}

void rleToBbox(const RLE *R, double *bb, unsigned long n) {
  unsigned long i;
  for (i = 0; i < n; i++) {
    unsigned int h, w, x, y, xs, ys, xe, ye, xp = 0, cc, t;
    unsigned long j, m;
    h = (unsigned int)R[i].h;
    w = (unsigned int)R[i].w;
    m = R[i].m;
    m = ((unsigned long)(m / 2)) * 2;
    xs = w;
    ys = h;
    xe = ye = 0;
    cc = 0;
    if (m == 0) {
      bb[4 * i + 0] = bb[4 * i + 1] = bb[4 * i + 2] = bb[4 * i + 3] = 0;
      continue;
    }
    for (j = 0; j < m; j++) {
      cc += R[i].cnts[j];
      t = cc - j % 2;
      y = t % h;
      x = (t - y) / h;
      if (j % 2 == 0)
        xp = x;
      else if (xp < x) {
        ys = 0;
        ye = h - 1;
      }
      xs = umin(xs, x);
      xe = umax(xe, x);
      ys = umin(ys, y);
      ye = umax(ye, y);
    }
    bb[4 * i + 0] = xs;
    bb[4 * i + 2] = xe - xs + 1;
    bb[4 * i + 1] = ys;
    bb[4 * i + 3] = ye - ys + 1;
  }
}

void rleIou(RLE *dt, RLE *gt, int m, int n, int *iscrowd, double *o) {
  int g, d;
  // double *db, *gb;
  int crowd;
  // db = (double *)malloc(sizeof(double) * m * 4);
  // rleToBbox(dt, db, m);
  // gb = (double *)malloc(sizeof(double) * n * 4);
  // rleToBbox(gt, gb, n);
  // bbIou(db, gb, m, n, iscrowd, o);
  // free(db);
  // free(gb);
  //#pragma omp parallel for num_threads(4)
  for (g = 0; g < n; g++)
#pragma omp parallel for num_threads(24)
    for (d = 0; d < m; d++)
      if (o[g * m + d] > 0) {
        crowd = iscrowd != NULL && iscrowd[g];
        if (dt[d].h != gt[g].h || dt[d].w != gt[g].w) {
          o[g * m + d] = -1;
          continue;
        }
        unsigned long ka, kb, a, b;
        uint c, ca, cb, ct, i, u;
        int va, vb;
        ca = dt[d].cnts[0];
        ka = dt[d].m;
        va = vb = 0;
        cb = gt[g].cnts[0];
        kb = gt[g].m;
        a = b = 1;
        i = u = 0;
        ct = 1;
        while (ct > 0) {
          c = umin(ca, cb);
          if (va || vb) {
            u += c;
            if (va && vb) i += c;
          }
          ct = 0;
          ca -= c;
          if (!ca && a < ka) {
            ca = dt[d].cnts[a++];
            va = !va;
          }
          ct += ca;
          cb -= c;
          if (!cb && b < kb) {
            cb = gt[g].cnts[b++];
            vb = !vb;
          }
          ct += cb;
        }
        if (i == 0)
          u = 1;
        else if (crowd)
          rleArea(dt + d, 1, &u);
        o[g * m + d] = (double)i / (double)u;
      }
}

void compute_iou(std::string iouType, int maxDet, int useCats) {
  assert(useCats > 0);

  //#pragma omp parallel for num_threads(96)
  for (size_t i = 0; i < imgids.size(); i++) {
    for (size_t c = 0; c < catids.size(); c++) {
      int catId = catids[c];
      int imgId = imgids[i];

      auto gtsm = &gts_map[key(imgId, catId)];
      auto dtsm = &dts_map[key(imgId, catId)];
      auto G = gtsm->id.size();
      auto D = dtsm->id.size();

      // py::tuple k = py::make_tuple(imgId, catId);

      if ((G == 0) && (D == 0)) {
        ious_map[key(imgId, catId)] = std::vector<double>();
        continue;
      }
      std::vector<float> scores;
      for (size_t i = 0; i < D; i++) {
        auto score = -dtsm->score[i];
        scores.push_back(score);
      }
      auto inds = sort_indices(scores);

      assert(iouType == "bbox" || iouType == "segm");

      if (iouType == "bbox") {
        std::vector<double> g;
        for (size_t i = 0; i < G; i++) {
          auto arr = gtsm->bbox[i];
          for (size_t j = 0; j < arr.size(); j++) {
            g.push_back((double)arr[j]);
          }
        }
        std::vector<double> d;
        for (size_t i = 0; i < std::min(D, (size_t)maxDet); i++) {
          auto arr = dtsm->bbox[inds[i]];
          for (size_t j = 0; j < arr.size(); j++) {
            d.push_back((double)arr[j]);
          }
        }
        // compute iou between each dt and gt region
        // iscrowd = [int(o['iscrowd']) for o in gt]
        std::vector<int> iscrowd(G);
        for (size_t i = 0; i < G; i++) {
          iscrowd[i] = gtsm->iscrowd[i];
        }
        int m = std::min(D, (size_t)maxDet);
        int n = G;
        if (m == 0 || n == 0) {
          ious_map[key(imgId, catId)] = std::vector<double>();
          continue;
        }
        std::vector<double> iou(m * n);
        // internal conversion from compressed RLE format to Python RLEs object
        // if (iouType == "bbox")
        bbIou(&d[0], &g[0], m, n, &iscrowd[0], &iou[0]);
        // rleIou(&dt[0],&gt[0],m,n,&iscrowd[0], double *o )
        ious_map[key(imgId, catId)] = iou;
      } else {
        std::vector<double> gb;
        for (size_t i = 0; i < G; i++) {
          auto arr = gtsm->bbox[i];
          for (size_t j = 0; j < arr.size(); j++) {
            gb.push_back((double)arr[j]);
          }
        }
        std::vector<double> db;
        for (size_t i = 0; i < std::min(D, (size_t)maxDet); i++) {
          auto arr = dtsm->bbox[inds[i]];
          for (size_t j = 0; j < arr.size(); j++) {
            db.push_back((double)arr[j]);
          }
        }

        std::vector<RLE> g(G);
        for (size_t i = 0; i < G; i++) {
          auto size = gtsm->segm_size[i];
          auto cnts = gtsm->segm_counts[i];
          rleInit(&g[i], size[0], size[1], cnts.size(), &cnts[0]);
        }
        std::vector<RLE> d(std::min(D, (size_t)maxDet));
        for (size_t i = 0; i < std::min(D, (size_t)maxDet); i++) {
          auto size = dtsm->segm_size[i];
          auto cnts = dtsm->segm_counts[inds[i]];
          rleInit(&d[i], size[0], size[1], cnts.size(), &cnts[0]);
        }
        std::vector<int> iscrowd(G);
        for (size_t i = 0; i < G; i++) {
          iscrowd[i] = gtsm->iscrowd[i];
        }
        int m = std::min(D, (size_t)maxDet);
        int n = G;
        if (m == 0 || n == 0) {
          ious_map[key(imgId, catId)] = std::vector<double>();
          for (size_t i = 0; i < g.size(); i++) {
            free(g[i].cnts);
          }
          for (size_t i = 0; i < d.size(); i++) {
            free(d[i].cnts);
          }
          continue;
        }
        std::vector<double> iou(m * n);
        // internal conversion from compressed RLE format to Python RLEs object
        // if (iouType == "bbox")
        bbIou(&db[0], &gb[0], m, n, &iscrowd[0], &iou[0]);
        rleIou(&d[0], &g[0], m, n, &iscrowd[0], &iou[0]);
        for (size_t i = 0; i < g.size(); i++) {
          free(g[i].cnts);
        }
        for (size_t i = 0; i < d.size(); i++) {
          free(d[i].cnts);
        }
        ious_map[key(imgId, catId)] = iou;
      }
    }
  }
}

std::string rleToString(const RLE *R) {
  /* Similar to LEB128 but using 6 bits/char and ascii chars 48-111. */
  unsigned long i, m = R->m, p = 0;
  long x;
  int more;
  char *s = (char *)malloc(sizeof(char) * m * 6);
  for (i = 0; i < m; i++) {
    x = (long)R->cnts[i];
    if (i > 2) x -= (long)R->cnts[i - 2];
    more = 1;
    while (more) {
      char c = x & 0x1f;
      x >>= 5;
      more = (c & 0x10) ? x != -1 : x != 0;
      if (more) c |= 0x20;
      c += 48;
      s[p++] = c;
    }
  }
  s[p] = 0;
  std::string str = std::string(s);
  free(s);
  return str;
}

std::string frUncompressedRLE(std::vector<int> cnts, std::vector<int> size,
                              int h, int w) {
  unsigned int *data =
      (unsigned int *)malloc(cnts.size() * sizeof(unsigned int));
  for (size_t i = 0; i < cnts.size(); i++) {
    data[i] = (unsigned int)cnts[i];
  }
  RLE R;  // = RLE(size[0],size[1],cnts.size(),data);
  R.h = size[0];
  R.w = size[1];
  R.m = cnts.size();
  R.cnts = data;
  std::string str = rleToString(&R);
  free(data);
  return str;
}

int uintCompare(const void *a, const void *b) {
  unsigned int c = *((unsigned int *)a), d = *((unsigned int *)b);
  return c > d ? 1 : c < d ? -1 : 0;
}

void rleFrPoly(RLE *R, const double *xy, int k, int h, int w) {
  /* upsample and get discrete points densely along entire boundary */
  int j, m = 0;
  double scale = 5;
  int *x, *y, *u, *v;
  unsigned int *a, *b;
  x = (int *)malloc(sizeof(int) * (k + 1));
  y = (int *)malloc(sizeof(int) * (k + 1));
  for (j = 0; j < k; j++) x[j] = (int)(scale * xy[j * 2 + 0] + .5);
  x[k] = x[0];
  for (j = 0; j < k; j++) y[j] = (int)(scale * xy[j * 2 + 1] + .5);
  y[k] = y[0];
  for (j = 0; j < k; j++)
    m += umax(abs(x[j] - x[j + 1]), abs(y[j] - y[j + 1])) + 1;
  u = (int *)malloc(sizeof(int) * m);
  v = (int *)malloc(sizeof(int) * m);
  m = 0;
  for (j = 0; j < k; j++) {
    int xs = x[j], xe = x[j + 1], ys = y[j], ye = y[j + 1], dx, dy, t, d;
    int flip;
    double s;
    dx = abs(xe - xs);
    dy = abs(ys - ye);
    flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
    if (flip) {
      t = xs;
      xs = xe;
      xe = t;
      t = ys;
      ys = ye;
      ye = t;
    }
    s = dx >= dy ? (double)(ye - ys) / dx : (double)(xe - xs) / dy;
    if (dx >= dy)
      for (d = 0; d <= dx; d++) {
        t = flip ? dx - d : d;
        u[m] = t + xs;
        v[m] = (int)(ys + s * t + .5);
        m++;
      }
    else
      for (d = 0; d <= dy; d++) {
        t = flip ? dy - d : d;
        v[m] = t + ys;
        u[m] = (int)(xs + s * t + .5);
        m++;
      }
  }
  /* get points along y-boundary and downsample */
  free(x);
  free(y);
  k = m;
  m = 0;
  double xd, yd;
  x = (int *)malloc(sizeof(int) * k);
  y = (int *)malloc(sizeof(int) * k);
  for (j = 1; j < k; j++)
    if (u[j] != u[j - 1]) {
      xd = (double)(u[j] < u[j - 1] ? u[j] : u[j] - 1);
      xd = (xd + .5) / scale - .5;
      if (floor(xd) != xd || xd < 0 || xd > w - 1) continue;
      yd = (double)(v[j] < v[j - 1] ? v[j] : v[j - 1]);
      yd = (yd + .5) / scale - .5;
      if (yd < 0)
        yd = 0;
      else if (yd > h)
        yd = h;
      yd = ceil(yd);
      x[m] = (int)xd;
      y[m] = (int)yd;
      m++;
    }
  /* compute rle encoding given y-boundary points */
  k = m;
  a = (unsigned int *)malloc(sizeof(unsigned int) * (k + 1));
  for (j = 0; j < k; j++) a[j] = (unsigned int)(x[j] * (int)(h) + y[j]);
  a[k++] = (unsigned int)(h * w);
  free(u);
  free(v);
  free(x);
  free(y);
  qsort(a, k, sizeof(unsigned int), uintCompare);
  unsigned int p = 0;
  for (j = 0; j < k; j++) {
    unsigned int t = a[j];
    a[j] -= p;
    p = t;
  }
  b = (unsigned int *)malloc(sizeof(unsigned int) * k);
  j = m = 0;
  b[m++] = a[j++];
  while (j < k)
    if (a[j] > 0)
      b[m++] = a[j++];
    else {
      j++;
      if (j < k) b[m - 1] += a[j++];
    }
  rleInit(R, h, w, m, b);
  free(a);
  free(b);
}

void rleMerge(const RLE *R, RLE *M, unsigned long n, int intersect) {
  unsigned int *cnts, c, ca, cb, cc, ct;
  int v, va, vb, vp;
  unsigned long i, a, b, h = R[0].h, w = R[0].w, m = R[0].m;
  RLE A, B;
  if (n == 0) {
    rleInit(M, 0, 0, 0, 0);
    return;
  }
  if (n == 1) {
    rleInit(M, h, w, m, R[0].cnts);
    return;
  }
  cnts = (unsigned int *)malloc(sizeof(unsigned int) * (h * w + 1));
  for (a = 0; a < m; a++) cnts[a] = R[0].cnts[a];
  for (i = 1; i < n; i++) {
    B = R[i];
    if (B.h != h || B.w != w) {
      h = w = m = 0;
      break;
    }
    rleInit(&A, h, w, m, cnts);
    ca = A.cnts[0];
    cb = B.cnts[0];
    v = va = vb = 0;
    m = 0;
    a = b = 1;
    cc = 0;
    ct = 1;
    while (ct > 0) {
      c = umin(ca, cb);
      cc += c;
      ct = 0;
      ca -= c;
      if (!ca && a < A.m) {
        ca = A.cnts[a++];
        va = !va;
      }
      ct += ca;
      cb -= c;
      if (!cb && b < B.m) {
        cb = B.cnts[b++];
        vb = !vb;
      }
      ct += cb;
      vp = v;
      if (intersect)
        v = va && vb;
      else
        v = va || vb;
      if (v != vp || ct == 0) {
        cnts[m++] = cc;
        cc = 0;
      }
    }
    rleFree(&A);
  }
  rleInit(M, h, w, m, cnts);
  free(cnts);
}

void rlesInit(RLE **R, unsigned long n) {
  unsigned long i;
  *R = (RLE *)malloc(sizeof(RLE) * n);
  for (i = 0; i < n; i++) rleInit((*R) + i, 0, 0, 0, 0);
}

std::vector<uint> frPoly(std::vector<std::vector<double>> poly, int h, int w) {
  size_t n = poly.size();
  RLE *Rs;
  rlesInit(&Rs, n);
  for (size_t i = 0; i < n; i++) {
    double *p = (double *)malloc(sizeof(double) * poly[i].size());
    for (size_t j = 0; j < poly[i].size(); j++) {
      p[j] = (double)poly[i][j];
    }
    rleFrPoly(&Rs[i], p, int(poly[i].size() / 2), h, w);
    free(p);
  }
  // _toString
  /*std::vector<char*> string;
  for (size_t i = 0; i < n; i++) {
    char* c_string = rleToString(&Rs[i]);
    string.push_back(c_string);
  }
  // _frString
  RLE *Gs;
  rlesInit(&Gs,n);
  for (size_t i = 0; i < n; i++) {
    rleFrString(&Gs[i],string[i],h,w);
  }*/
  // merge(rleObjs, intersect=0)
  RLE R;
  int intersect = 0;
  rleMerge(Rs, &R, n, intersect);
  // std::string str = rleToString(&R);
  // for (size_t i = 0; i < n; i++) {
  //   free(Rs[i].cnts);
  // }
  std::vector<uint> cnts(R.cnts, R.cnts + R.m);
  free(Rs);
  // return str;
  return cnts;
}

unsigned int area(std::vector<int> &size, std::vector<uint> &counts) {
  // _frString
  RLE *Rs;
  rlesInit(&Rs, 1);
  // char *str = new char[counts.length() + 1];
  // strcpy(str, counts.c_str());
  // rleFrString(&Rs[0], str, size[0], size[1]);
  // delete[] str;
  rleInit(Rs, size[0], size[1], counts.size(), &counts[0]);
  unsigned int a;
  rleArea(Rs, 1, &a);
  for (size_t i = 0; i < 1; i++) {
    free(Rs[i].cnts);
  }
  free(Rs);
  return a;
}

std::vector<float> toBbox(std::vector<int> &size, std::vector<uint> &counts) {
  // _frString
  RLE *Rs;
  rlesInit(&Rs, 1);
  // char *str = new char[counts.length() + 1];
  // strcpy(str, counts.c_str());
  // rleFrString(&Rs[0], str, size[0], size[1]);
  // delete[] str;

  rleInit(Rs, size[0], size[1], counts.size(), &counts[0]);

  std::vector<double> bb(4 * 1);
  rleToBbox(Rs, &bb[0], 1);
  std::vector<float> bbf(bb.size());
  for (size_t i = 0; i < bb.size(); i++) {
    bbf[i] = (float)bb[i];
  }
  for (size_t i = 0; i < 1; i++) {
    free(Rs[i].cnts);
  }
  free(Rs);
  return bbf;
}

void annToRLE(anns_struct &ann, std::vector<std::vector<int>> &size,
              std::vector<std::vector<uint>> &counts, int h, int w) {
  auto is_segm_list = ann.segm_list.size() > 0;
  auto is_cnts_list = is_segm_list ? 0 : ann.segm_counts_list.size() > 0;
  if (is_segm_list) {
    std::vector<int> segm_size{h, w};
    auto cnts = ann.segm_list;
    auto segm_counts = frPoly(cnts, h, w);
    size.push_back(segm_size);
    counts.push_back(segm_counts);
  } else if (is_cnts_list) {
    auto segm_size = ann.segm_size;
    auto cnts = ann.segm_counts_list;
    // auto segm_counts = frUncompressedRLE(cnts, segm_size, h, w);
    size.push_back(segm_size);
    counts.push_back(cnts);
  } else {
    // auto segm_size = ann.segm_size;
    // auto segm_counts = ann.segm_counts_str;
    // size.push_back(segm_size);
    // counts.push_back(segm_counts);
    // std::cout << "\n\n\n\n\n WE SHOULD NOT BE HERE \n\n\n\n\n";
  }
}

void getAnnsIds(std::unordered_map<size_t, std::vector<anns_struct>> &imgToAnns,
                std::unordered_map<size_t, anns_struct> &anns,
                std::vector<int64_t> &ids, std::vector<int64_t> &imgIds,
                std::vector<int64_t> &catIds) {
  for (size_t i = 0; i < imgIds.size(); i++) {
    auto hasimg = imgToAnns.find(imgIds[i]) != imgToAnns.end();
    if (hasimg) {
      auto anns = imgToAnns[imgIds[i]];
      for (size_t j = 0; j < anns.size(); j++) {
        //       auto catid = anns[j].category_id;
        //        auto hascat = (std::find(catIds.begin(), catIds.end(), catid)
        //        != catIds.end());  // set might be faster? does it matter? if
        //        (hascat) {
        // auto area = py::cast<float>(anns[j]["area"]);
        // some indices can have float values, so cast to double first
        ids.push_back(anns[j].id);
        //        }
      }
    }
  }
}

void cpp_load_res_numpy(py::dict dataset,
                        std::vector<std::vector<float>> data) {
  /*void cpp_load_res_numpy(py::dict dataset, py::array data) {
  auto buf = data.request();
  //float* data_ptr = (float*)buf.ptr;
  double* data_ptr = (double*)buf.ptr;//sometimes predictions are in double?
  size_t size = buf.shape[0];*/
  for (size_t i = 0; i < data.size(); i++) {
    /*  for (size_t i = 0; i < size; i++) {
    auto datai = &data_ptr[i*7];*/
    anns_struct ann;
    ann.image_id = int(data[i][0]);
    // ann.image_id = int(datai[0]);
    ann.bbox =
        std::vector<float>{data[i][1], data[i][2], data[i][3], data[i][4]};
    // ann.bbox = std::vector<float>{(float)datai[1], (float)datai[2],
    // (float)datai[3], (float)datai[4]};
    ann.score = data[i][5];
    // ann.score = datai[5];
    ann.category_id = data[i][6];
    // ann.category_id = datai[6];
    auto bb = ann.bbox;
    auto x1 = bb[0];
    auto x2 = bb[0] + bb[2];
    auto y1 = bb[1];
    auto y2 = bb[1] + bb[3];
    ann.segm_list =
        std::vector<std::vector<double>>{{x1, y1, x1, y2, x2, y2, x2, y1}};
    ann.area = bb[2] * bb[3];
    ann.id = i + 1;
    ann.iscrowd = 0;

    auto k = key(ann.image_id, ann.category_id);
    data_struct *tmp = &dts_map[k];
    tmp->area.push_back(ann.area);
    tmp->iscrowd.push_back(ann.iscrowd);
    tmp->bbox.push_back(ann.bbox);
    tmp->score.push_back(ann.score);
    tmp->id.push_back(ann.id);
  }
}

void cpp_load_res_segm(py::dict dataset, std::vector<std::vector<float>> anns,
                       std::vector<std::string> cnts) {
  std::vector<std::vector<uint>> cnts_uint(anns.size());
#pragma omp parallel for num_threads(24)
  for (size_t i = 0; i < anns.size(); ++i) {
    auto segm_str = cnts[i];
    char *val = new char[segm_str.length() + 1];
    auto segm_size = std::vector<int>{anns[i][7], anns[i][8]};
    strcpy(val, segm_str.c_str());
    std::vector<uint> tmp;
    cntsFrString(tmp, val, segm_size[0], segm_size[1]);
    cnts_uint[i] = tmp;
  }

  for (size_t i = 0; i < anns.size(); ++i) {
    anns_struct ann;
    ann.image_id = int(anns[i][0]);
    ann.category_id = int64_t(anns[i][6]);
    // now only support compressed RLE format as segmentation results
    ann.segm_size = std::vector<int>{anns[i][7], anns[i][8]};
    auto segm_str = cnts[i];
    ann.segm_counts_list = cnts_uint[i];
    ann.bbox =
        std::vector<float>{anns[i][9], anns[i][10], anns[i][11], anns[i][12]};

    ann.score = anns[i][5];

    ann.id = i + 1;
    ann.iscrowd = 0;
    auto k = key(ann.image_id, ann.category_id);
    data_struct *tmp = &dts_map[k];
    tmp->area.push_back(ann.area);
    tmp->iscrowd.push_back(ann.iscrowd);
    tmp->bbox.push_back(ann.bbox);
    tmp->score.push_back(ann.score);
    tmp->id.push_back(ann.id);

    // convert ground truth to mask if iouType == 'segm'
    auto h = imgsgt[(size_t)ann.image_id]
                 .h;  // We pass in the dataset from the annotations anyways
    auto w = imgsgt[(size_t)ann.image_id].w;
    annToRLE(ann, tmp->segm_size, tmp->segm_counts, h, w);
  }
}

void cpp_load_res(py::dict dataset, std::vector<py::dict> anns) {
  auto isbbox = anns[0].contains("bbox") &&
                (py::cast<std::vector<float>>(anns[0]["bbox"]).size() > 0);
  // assert(!iscaption && (isbbox || issegm));
  assert(isbbox);

  if (isbbox) {
    for (size_t i = 0; i < anns.size(); i++) {
      anns_struct ann;
      ann.image_id = py::cast<int>(anns[i]["image_id"]);
      ann.category_id = py::cast<int64_t>(anns[i]["category_id"]);
      auto bb = py::cast<std::vector<float>>(anns[i]["bbox"]);
      ann.bbox = bb;
      // auto x1 = bb[0];
      // auto x2 = bb[0] + bb[2];
      // auto y1 = bb[1];
      // auto y2 = bb[1] + bb[3];
      // if (!issegm) {
      //   ann.segm_list =
      //       std::vector<std::vector<double>>{{x1, y1, x1, y2, x2, y2, x2,
      //       y1}};
      // } else {  // do we need all of these?
      //   auto is_segm_list =
      //   py::isinstance<py::list>(anns[i]["segmentation"]); auto
      //   is_cnts_list
      //   =
      //       is_segm_list
      //           ? 0
      //           :
      //           py::isinstance<py::list>(anns[i]["segmentation"]["counts"]);
      //   if (is_segm_list) {
      //     ann.segm_list = py::cast<std::vector<std::vector<double>>>(
      //         anns[i]["segmentation"]);
      //   } else if (is_cnts_list) {
      //     ann.segm_size =
      //         py::cast<std::vector<int>>(anns[i]["segmentation"]["size"]);
      //     ann.segm_counts_list =
      //         py::cast<std::vector<int>>(anns[i]["segmentation"]["counts"]);
      //   } else {
      //     ann.segm_size =
      //         py::cast<std::vector<int>>(anns[i]["segmentation"]["size"]);
      //     ann.segm_counts_str =
      //         py::cast<std::string>(anns[i]["segmentation"]["counts"]);
      //   }
      // }
      ann.score = py::cast<float>(anns[i]["score"]);
      ann.area = bb[2] * bb[3];
      ann.id = i + 1;
      ann.iscrowd = 0;
      // annsdt[ann.id] = ann;
      // dtimgToAnns[(size_t)ann.image_id].push_back(ann);
      auto k = key(ann.image_id, ann.category_id);
      data_struct *tmp = &dts_map[k];
      tmp->area.push_back(ann.area);
      tmp->iscrowd.push_back(ann.iscrowd);
      tmp->bbox.push_back(ann.bbox);
      tmp->score.push_back(ann.score);
      tmp->id.push_back(ann.id);
    }
  } else {
    // Removing this, This is only needed if we custom set the annonataions,
    // but the imgs should already be loaded by creating the index
    // std::unordered_map<size_t, image_struct> imgsdt;
    // auto imgs = py::cast<std::vector<py::dict>>(dataset["images"]);
    // for (size_t i = 0; i < imgs.size(); i++) {
    //   image_struct img;
    //   img.id = (size_t)py::cast<double>(imgs[i]["id"]);
    //   img.h = py::cast<int>(imgs[i]["height"]);
    //   img.w = py::cast<int>(imgs[i]["width"]);
    //   imgsdt[img.id] = img;
    // }
    for (size_t i = 0; i < anns.size(); i++) {
      anns_struct ann;
      ann.image_id = py::cast<int>(anns[i]["image_id"]);
      ann.category_id = py::cast<int64_t>(anns[i]["category_id"]);
      // now only support compressed RLE format as segmentation results
      ann.segm_size =
          py::cast<std::vector<int>>(anns[i]["segmentation"]["size"]);
      auto segm_str = anns[i]["segmentation"]["counts"].cast<std::string>();

      // ann.segm_counts_list =
      //    anns[i]["segmentation"]["counts"].cast<std::vector<uint>>();
      char *val = new char[segm_str.length() + 1];
      strcpy(val, segm_str.c_str());
      cntsFrString(ann.segm_counts_list, val, ann.segm_size[0],
                   ann.segm_size[1]);
      // auto test = anns[i]["segmentation"]["counts"].unchecked<1>();
      // ann.segm_counts_list =
      //    std::vector<uint>(test.data(), test.data() + test.size());
      // py::cast<std::vector<uint>>(anns[i]["segmentation"]["counts"]);
      //  .cast<py::array>()
      //  .cast<std::vector<uint>>();
      // py::array_t<uint32_t> casted_array =
      //     py::cast<py::array>(anns[i]["segmentation"]["counts"]);

      // ann.segm_counts_list = casted_array;
      // std::cout << ann.segm_counts_list.size() << "\t";
      // ann.area = area(ann.segm_size, ann.segm_counts_list);

      auto bb = py::cast<std::vector<float>>(anns[i]["segmentation"]["bbox"]);
      ann.bbox = bb;

      ann.score = py::cast<float>(anns[i]["score"]);
      ann.id = i + 1;
      ann.iscrowd = 0;
      // annsdt[ann.id] = ann;
      // dtimgToAnns[(size_t)ann.image_id].push_back(ann);
      auto k = key(ann.image_id, ann.category_id);
      data_struct *tmp = &dts_map[k];
      tmp->area.push_back(ann.area);
      tmp->iscrowd.push_back(ann.iscrowd);
      tmp->bbox.push_back(ann.bbox);
      tmp->score.push_back(ann.score);
      tmp->id.push_back(ann.id);
      // convert ground truth to mask if iouType == 'segm'
      auto h = imgsgt[(size_t)ann.image_id]
                   .h;  // We pass in the dataset from the annotations anyways
      auto w = imgsgt[(size_t)ann.image_id].w;
      annToRLE(ann, tmp->segm_size, tmp->segm_counts, h, w);
    }
  }
}

void cpp_create_index(py::dict dataset) {
  if (imgsgt.size() > 0 && imgids.size() > 0 && catids.size() > 0) {
    printf("GT annotations already exist!\n");
    return;
    // clear arrays
    /*printf("GT annotations already exist, cleanup and create again...\n");
    std::unordered_map<size_t, data_struct>().swap(gts_map);
    std::unordered_map<size_t, image_struct>().swap(imgsgt);
    std::vector<int64_t>().swap(imgids);
    std::vector<int64_t>().swap(catids);*/
  }

  auto imgs = py::cast<std::vector<py::dict>>(dataset["images"]);
  for (size_t i = 0; i < imgs.size(); i++) {
    image_struct img;
    img.id = (size_t)py::cast<double>(imgs[i]["id"]);
    img.h = py::cast<int>(imgs[i]["height"]);
    img.w = py::cast<int>(imgs[i]["width"]);
    imgsgt[img.id] = img;
    imgids.push_back(img.id);
  }
  auto cats = py::cast<std::vector<py::dict>>(dataset["categories"]);
  for (size_t i = 0; i < cats.size(); i++) {
    auto catid = py::cast<int>(cats[i]["id"]);
    catids.push_back(catid);
  }

  auto anns = py::cast<std::vector<py::dict>>(dataset["annotations"]);
  for (size_t i = 0; i < anns.size(); i++) {
    anns_struct ann;
    ann.image_id = py::cast<int>(anns[i]["image_id"]);
    ann.category_id = py::cast<int64_t>(anns[i]["category_id"]);
    ann.id = (size_t)py::cast<double>(anns[i]["id"]);
    ann.area = py::cast<float>(anns[i]["area"]);
    ann.iscrowd = py::cast<int>(anns[i]["iscrowd"]);
    /*auto has_score = (anns[i].contains("score"));
    if (has_score) {
      ann.score = py::cast<float>(anns[i]["score"]);
    }*/
    ann.bbox = py::cast<std::vector<float>>(anns[i]["bbox"]);

    auto is_segm_list = py::isinstance<py::list>(anns[i]["segmentation"]);
    auto is_cnts_list =
        is_segm_list
            ? 0
            : py::isinstance<py::list>(anns[i]["segmentation"]["counts"]);

    if (is_segm_list) {
      ann.segm_list =
          py::cast<std::vector<std::vector<double>>>(anns[i]["segmentation"]);
    } else if (is_cnts_list) {
      ann.segm_size =
          py::cast<std::vector<int>>(anns[i]["segmentation"]["size"]);
      ann.segm_counts_list =
          py::cast<std::vector<uint>>(anns[i]["segmentation"]["counts"]);
    } else {
      ann.segm_size =
          py::cast<std::vector<int>>(anns[i]["segmentation"]["size"]);
      ann.segm_counts_str =
          py::cast<std::string>(anns[i]["segmentation"]["counts"]);
    }

    auto k = key(ann.image_id, ann.category_id);
    data_struct *tmp = &gts_map[k];
    tmp->area.push_back(ann.area);
    tmp->iscrowd.push_back(ann.iscrowd);
    tmp->bbox.push_back(ann.bbox);
    tmp->ignore.push_back(ann.iscrowd != 0);
    // tmp->score.push_back(ann.score);
    tmp->id.push_back(ann.id);
    auto h = imgsgt[(size_t)ann.image_id].h;
    auto w = imgsgt[(size_t)ann.image_id].w;
    annToRLE(ann, tmp->segm_size, tmp->segm_counts, h, w);
  }
}

PYBIND11_MODULE(ext, m) {
  m.doc() = "pybind11 pycocotools plugin";
  m.def("cpp_evaluate", &cpp_evaluate, "");
  m.def("cpp_evaluate_dist", &cpp_evaluate_dist, "");
  m.def("cpp_create_index", &cpp_create_index, "");
  m.def("cpp_load_res", &cpp_load_res, "");
  m.def("cpp_load_res_segm", &cpp_load_res_segm, "");
  m.def("cpp_load_res_numpy", &cpp_load_res_numpy, "");
  pybind11::bind_vector<std::vector<uint>>(m, "CountsVec");
  py::implicitly_convertible<py::list, std::vector<uint>>();
}
