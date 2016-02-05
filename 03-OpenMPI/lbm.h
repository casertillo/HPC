#ifndef LBM_HDR_FILE
#define LBM_HDR_FILE

#define NSPEEDS         9

/* Size of box in imaginary 'units */
#define BOX_X_SIZE (100.0)
#define BOX_Y_SIZE (100.0)

/* struct to hold the parameter values */
typedef struct {
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    max_iters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
  int tot_cells;
} param_t;

/* struct to hold the 'speed' values */
typedef struct {
  float speeds[NSPEEDS];
} speed_t;

typedef enum { ACCEL_ROW, ACCEL_COLUMN } accel_e;
typedef struct {
    accel_e col_or_row;
    int idx;
} accel_area_t;

/* obstacle positions */
typedef struct {
    float obs_x_min;
    float obs_x_max;
    float obs_y_min;
    float obs_y_max;
} obstacle_t;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
void initialise(const char* params_file, accel_area_t * accel_area,param_t* params, speed_t** cells_ptr, speed_t** tmp_cells_ptr, 
           int** obstacles_ptr, float** av_vels_ptr, int** accelgrid_ptr);

void accelerate_flow(const param_t params,const accel_area_t accel_area, speed_t* cells, int start_row, int end_row, int* accelgrid);
void lattice(const param_t params, speed_t* cells, speed_t* tmp_cells, int* obstacles, float* av_vels, int start_row, int end_row, int stepnum);
void write_values(const param_t params, speed_t* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
void finalise(const param_t* params, speed_t** cells_ptr, speed_t** tmp_cells_ptr,
         int** obstacles_ptr, float** av_vels_ptr, int** accelgrid_ptr, int** rowSetup);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const param_t params, speed_t* cells);

/* calculate Reynolds number */
float calc_reynolds(const param_t params, speed_t* cells, int* obstacles, float final_average_velocity);

void calc_row_setup(int** rowSetup_ptr, int rank, int halorow, int extra, param_t params);

/* utility functions */
#define DIE(...) exit_with_error(__LINE__, __FILE__, __VA_ARGS__)
void exit_with_error(int line, const char* filename, const char* format, ...)
__attribute__ ((format (printf, 3, 4)));

#endif