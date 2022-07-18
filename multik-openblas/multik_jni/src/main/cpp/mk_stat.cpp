#include "mk_stat.h"
#include "algorithm"

double array_median(const void *arr, int size, int type) {
  double ret;
  int mid = size / 2;
  switch (type) {
	case 1: {
	  auto *sorted = (int8_t *)arr;
	  std::sort(sorted, sorted + size);
	  if (size % 2 != 0) {
		ret = sorted[mid];
	  } else {
		ret = (static_cast<double>(sorted[mid - 1]) + static_cast<double>(sorted[mid])) / 2.0;
	  }
	  break;
	}
	case 2: {
	  auto *sorted = (int16_t *)arr;
	  std::sort(sorted, sorted + size);
	  if (size % 2 != 0) {
		ret = sorted[mid];
	  } else {
		ret = (static_cast<double>(sorted[mid - 1]) + static_cast<double>(sorted[mid])) / 2.0;
	  }
	  break;
	}
	case 3: {
	  auto *sorted = (int32_t *)arr;
	  std::sort(sorted, sorted + size);
	  if (size % 2 != 0) {
		ret = sorted[mid];
	  } else {
		ret = (static_cast<double>(sorted[mid - 1]) + static_cast<double>(sorted[mid])) / 2.0;
	  }
	  break;
	}
	case 4: {
	  auto *sorted = (int64_t *)arr;
	  std::sort(sorted, sorted + size);
	  if (size % 2 != 0) {
		ret = sorted[mid];
	  } else {
		ret = (static_cast<double>(sorted[mid - 1]) + static_cast<double>(sorted[mid])) / 2.0;
	  }
	  break;
	}
	case 5: {
	  auto *sorted = (float *)arr;
	  std::sort(sorted, sorted + size);
	  if (size % 2 != 0) {
		ret = sorted[mid];
	  } else {
		ret = (static_cast<double>(sorted[mid - 1]) + static_cast<double>(sorted[mid])) / 2.0;
	  }
	  break;
	}
	case 6: {
	  auto *sorted = (double *)arr;
	  std::sort(sorted, sorted + size);
	  if (size % 2 != 0) {
		ret = sorted[mid];
	  } else {
		ret = (sorted[mid - 1] + sorted[mid]) / 2.0;
	  }
	  break;
	}
	default:break;
  }
  return ret;
}

