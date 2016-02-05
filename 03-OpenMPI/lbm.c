
/*
** code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the bhatnagar-gross-krook collision step.
**
** the 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** a 2d grid:
**
**           cols
**       --- --- ---
**      | d | e | f |
** rows  --- --- ---
**      | a | b | c |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1d array:
**
**  --- --- --- --- --- ---
** | a | b | c | d | e | f |
**  --- --- --- --- --- ---
**
** grid indicies are:
**
**          ny
**          ^       cols(jj)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(ii) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   d2q9-bgk.exe input.params obstacles.dat
**
** be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "mpi.h"

#include <math.h>
#include <unistd.h>
#include <ctype.h>
#include <string.h>
#include <stdarg.h>
#include <getopt.h>
#include "lbm.h"

int main(int argc, char* argv[])
{
    param_t  params;              /* struct to hold parameter values */
    speed_t* cells     = NULL;    /* grid containing fluid densities */
    speed_t* tmp_cells = NULL;    /* scratch space */
    int*     obstacles = NULL;    /* grid indicating which cells are blocked */
    int*     rowsSetup = NULL;
    int*     accelgrid = NULL;
    float*   av_vels   = NULL;    /* a record of the av. velocity computed for each timestep */
    int      ii, rank, size, tag=0, jj;      /* generic counter */
    struct timeval timstr;        /* structure to hold elapsed time */
    struct rusage ru;             /* structure to hold CPU time--system and user */
    double tic,toc;               /* floating point numbers to calculate elapsed wallclock time */
    double usrtim;                /* floating point number to record elapsed user CPU time */
    double systim;                /* floating point number to record elapsed system CPU time */

    int halorow;
    int buff;
    int extra;
    int start_row;
    int end_row;
    MPI_Status status;
    accel_area_t accel_area;

    /* initialise our data structures and load values from file */
    initialise(argv[1], &accel_area, &params, &cells, &tmp_cells, &obstacles, &av_vels, &accelgrid);

    // Initialize MPI environment.
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	extra = params.ny%size;

    halorow = (rank<extra) ? (params.ny/size + 1) * params.nx : (params.ny/size) * params.nx;

    calc_row_setup(&rowsSetup, rank, halorow, extra, params);

    buff = rowsSetup[0];
    start_row = rowsSetup[1];
    end_row = rowsSetup[2];

    /* iterate for max_iters timesteps */
    gettimeofday(&timstr,NULL);
    tic=timstr.tv_sec+(timstr.tv_usec/1000000.0);

    for(ii=0;ii<params.max_iters;ii++) {

        accelerate_flow(params,accel_area,cells,start_row, end_row, accelgrid);
        lattice(params,cells,tmp_cells,obstacles, av_vels, start_row, end_row, ii);   
    }

    gettimeofday(&timstr,NULL);
    toc=timstr.tv_sec+(timstr.tv_usec/1000000.0);
    getrusage(RUSAGE_SELF, &ru);
    timstr=ru.ru_utime;        
    usrtim=timstr.tv_sec+(timstr.tv_usec/1000000.0);
    timstr=ru.ru_stime;        
    systim=timstr.tv_sec+(timstr.tv_usec/1000000.0);

    float* buffer = malloc(buff * 9 * sizeof(float));


    if(rank != 0) {

        for(ii=0;ii<halorow;ii++) {
            buffer[9*ii]   = cells[start_row+ii].speeds[0];
            buffer[9*ii+1] = cells[start_row+ii].speeds[1];
            buffer[9*ii+2] = cells[start_row+ii].speeds[2];
            buffer[9*ii+3] = cells[start_row+ii].speeds[3];
            buffer[9*ii+4] = cells[start_row+ii].speeds[4];
            buffer[9*ii+5] = cells[start_row+ii].speeds[5];
            buffer[9*ii+6] = cells[start_row+ii].speeds[6];
            buffer[9*ii+7] = cells[start_row+ii].speeds[7];
            buffer[9*ii+8] = cells[start_row+ii].speeds[8];
        }
        MPI_Send(buffer, 9*buff, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
    } 
    else {

        if(extra == 0) {
        	for(ii=1;ii<size;ii++) {
		        MPI_Recv(buffer, 9*buff, MPI_FLOAT, ii, tag, MPI_COMM_WORLD, &status);
		        for(jj=0;jj<halorow;jj++) {
		            cells[halorow*ii+jj].speeds[0] = buffer[9*jj];
		            cells[halorow*ii+jj].speeds[1] = buffer[9*jj+1];
		            cells[halorow*ii+jj].speeds[2] = buffer[9*jj+2];
		            cells[halorow*ii+jj].speeds[3] = buffer[9*jj+3];
		            cells[halorow*ii+jj].speeds[4] = buffer[9*jj+4];
		            cells[halorow*ii+jj].speeds[5] = buffer[9*jj+5];
		            cells[halorow*ii+jj].speeds[6] = buffer[9*jj+6];
		            cells[halorow*ii+jj].speeds[7] = buffer[9*jj+7];
		            cells[halorow*ii+jj].speeds[8] = buffer[9*jj+8];
		        }
        	}
        } else {
		    for(ii=1;ii<extra;ii++) {
		        MPI_Recv(buffer, 9*buff, MPI_FLOAT, ii, tag, MPI_COMM_WORLD, &status);
		        for(jj=0;jj<halorow;jj++) {
		            cells[halorow*ii+jj].speeds[0] = buffer[9*jj];
		            cells[halorow*ii+jj].speeds[1] = buffer[9*jj+1];
		            cells[halorow*ii+jj].speeds[2] = buffer[9*jj+2];
		            cells[halorow*ii+jj].speeds[3] = buffer[9*jj+3];
		            cells[halorow*ii+jj].speeds[4] = buffer[9*jj+4];
		            cells[halorow*ii+jj].speeds[5] = buffer[9*jj+5];
		            cells[halorow*ii+jj].speeds[6] = buffer[9*jj+6];
		            cells[halorow*ii+jj].speeds[7] = buffer[9*jj+7];
		            cells[halorow*ii+jj].speeds[8] = buffer[9*jj+8];
		        }
		    }
		    for(ii=extra;ii<size;ii++) {
		        MPI_Recv(buffer, 9*buff, MPI_FLOAT, ii, tag, MPI_COMM_WORLD, &status);
		        for(jj=0;jj<(params.ny/size) * params.nx;jj++) {
              int local_extra = halorow * extra + (halorow-params.nx)* (ii-extra);
              cells[local_extra + jj].speeds[0] = buffer[9*jj];
              cells[local_extra + jj].speeds[1] = buffer[9*jj+1];
              cells[local_extra + jj].speeds[2] = buffer[9*jj+2];
		          cells[local_extra + jj].speeds[3] = buffer[9*jj+3];
		          cells[local_extra + jj].speeds[4] = buffer[9*jj+4];
		          cells[local_extra + jj].speeds[5] = buffer[9*jj+5];
		          cells[local_extra + jj].speeds[6] = buffer[9*jj+6];
		          cells[local_extra + jj].speeds[7] = buffer[9*jj+7];
		          cells[local_extra + jj].speeds[8] = buffer[9*jj+8];
		        }
		    }
        }
        free(buffer);
        buffer = NULL;
    }
    
    MPI_Finalize(); /*Finilize MPI*/

    if(rank == 0) {
        printf("==done==\n");
        printf("Reynolds number:\t\t%.12E\n",calc_reynolds(params,cells,obstacles,av_vels[params.max_iters-1]));
        printf("Elapsed time:\t\t\t%.6lf (s)\n", toc-tic);
        printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
        printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
        write_values(params,cells,obstacles,av_vels);
        finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels, &accelgrid, &rowsSetup);
    }
    
    return EXIT_SUCCESS;
}

void initialise(const char* params_file, accel_area_t * accel_area ,
           param_t* params, speed_t** cells_ptr, speed_t** tmp_cells_ptr, 
           int** obstacles_ptr, float** av_vels_ptr, int** accelgrid_ptr)
{
  FILE   *fp;            /* file pointer */
  int    ii,jj, kk;          /* generic counters */
  int    retval;         /* to hold return value for checking */
  float w0,w1,w2;       /* weighting factors */

  /* Rectangular obstacles */
  int n_obstacles;
  obstacle_t * obstacles;

  /* open the parameter file */
    fp = fopen(params_file, "r");

    if (NULL == fp)
    {
        DIE("Unable to open param file %s", params_file);
    }

    /* read in the parameter values */
    retval = fscanf(fp,"%d\n",&(params->nx));
    if (retval != 1) DIE("Could not read param file: nx");
    retval = fscanf(fp,"%d\n",&(params->ny));
    if (retval != 1) DIE("Could not read param file: ny");
    retval = fscanf(fp,"%d\n",&(params->max_iters));
    if (retval != 1) DIE("Could not read param file: max_iters");
    retval = fscanf(fp,"%d\n",&(params->reynolds_dim));
    if (retval != 1) DIE("Could not read param file: reynolds_dim");
    retval = fscanf(fp,"%f\n",&(params->density));
    if (retval != 1) DIE("Could not read param file: density");
    retval = fscanf(fp,"%f\n",&(params->accel));
    if (retval != 1) DIE("Could not read param file: accel");
    retval = fscanf(fp,"%f\n",&(params->omega));
    if (retval != 1) DIE("Could not read param file: omega");

    if (params->nx < 100) DIE("x dimension of grid in input file was too small (must be >100)");
    if (params->ny < 100) DIE("y dimension of grid in input file was too small (must be >100)");

    /* read column/row to accelerate */
    char accel_dir_buf[11];
    int idx;
    retval = fscanf(fp,"%*s %10s %d\n", accel_dir_buf, &idx);
    if (retval != 2) DIE("Could not read param file: could not parse acceleration specification");
    if (idx > 100 || idx < 0) DIE("Acceleration index (%d) out of range (must be bigger than 0 and less than 100)", idx);

    if (!(strcmp(accel_dir_buf, "row")))
    {
        accel_area->col_or_row = ACCEL_ROW;
        accel_area->idx = idx*(params->ny/BOX_Y_SIZE);
    }
    else if (!(strcmp(accel_dir_buf, "column")))
    {
        accel_area->col_or_row = ACCEL_COLUMN;
        accel_area->idx = idx*(params->nx/BOX_X_SIZE);
    }
    else
    {
        DIE("Error reading param file: Unexpected acceleration specification '%s'", accel_dir_buf);
    }

    /* read obstacles */
    retval = fscanf(fp, "%d %*s\n", &n_obstacles);
    if (retval != 1) DIE("Could not read param file: n_obstacles");
    obstacles = (obstacle_t*) malloc(sizeof(obstacle_t)*(n_obstacles));
    
    for (ii = 0; ii < n_obstacles; ii++)
    {
        retval = fscanf(fp,"%f %f %f %f\n",
            &obstacles[ii].obs_x_min, &obstacles[ii].obs_y_min,
            &obstacles[ii].obs_x_max, &obstacles[ii].obs_y_max);
        if (retval != 4) DIE("Could not read param file: location of obstacle %d", ii + 1);
        if (obstacles[ii].obs_x_min < 0 || obstacles[ii].obs_y_min < 0 ||
            obstacles[ii].obs_x_max > 100 || obstacles[ii].obs_y_max > 100)
        {
            DIE("Obstacle %d out of range (must be bigger than 0 and less than 100)", ii);
        }
        if (obstacles[ii].obs_x_min > obstacles[ii].obs_x_max) DIE("Left x coordinate is bigger than right x coordinate - this will result in no obstacle being made");
        if (obstacles[ii].obs_y_min > obstacles[ii].obs_y_max) DIE("Bottom y coordinate is bigger than top y coordinate - this will result in no obstacle being made");
    }

  /* and close up the file */
  fclose(fp);

  /* main grid */
    /* Allocate arrays */
    *cells_ptr = (speed_t*) malloc(sizeof(speed_t)*(params->ny*params->nx));
    if (*cells_ptr == NULL) DIE("Cannot allocate memory for cells");

    *tmp_cells_ptr = (speed_t*) malloc(sizeof(speed_t)*(params->ny*params->nx));
    if (*tmp_cells_ptr == NULL) DIE("Cannot allocate memory for tmp_cells");

    *obstacles_ptr = (int*) malloc(sizeof(int)*(params->ny*params->nx));
    if (*obstacles_ptr == NULL) DIE("Cannot allocate memory for patches");

    *av_vels_ptr = (float*) malloc(sizeof(float)*(params->max_iters));
    if (*av_vels_ptr == NULL) DIE("Cannot allocate memory for av_vels");

  /* initialise densities */
  w0 = params->density * 4.0/9.0;
  w1 = params->density      /9.0;
  w2 = params->density      /36.0;

  for(ii=0;ii<params->ny;ii++) {
    for(jj=0;jj<params->nx;jj++) {
      /* centre */
      (*cells_ptr)[ii*params->nx + jj].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii*params->nx + jj].speeds[1] = w1;
      (*cells_ptr)[ii*params->nx + jj].speeds[2] = w1;
      (*cells_ptr)[ii*params->nx + jj].speeds[3] = w1;
      (*cells_ptr)[ii*params->nx + jj].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii*params->nx + jj].speeds[5] = w2;
      (*cells_ptr)[ii*params->nx + jj].speeds[6] = w2;
      (*cells_ptr)[ii*params->nx + jj].speeds[7] = w2;
      (*cells_ptr)[ii*params->nx + jj].speeds[8] = w2;

      (*obstacles_ptr)[ii*params->nx + jj] = 0;
    }
  }
    int obs = 0;
    /* Fill in locations of obstacles */
    for (ii = 0; ii < params->ny; ii++)
    {
        for (jj = 0; jj < params->nx; jj++)
        {
            /* coordinates of (jj, ii) scaled to 'real world' terms */
            const float x_pos = jj*(BOX_X_SIZE/params->nx);
            const float y_pos = ii*(BOX_Y_SIZE/params->ny);

            for (kk = 0; kk < n_obstacles; kk++)
            {
                if (x_pos >= obstacles[kk].obs_x_min &&
                    x_pos <  obstacles[kk].obs_x_max &&
                    y_pos >= obstacles[kk].obs_y_min &&
                    y_pos <  obstacles[kk].obs_y_max)
                {
                    (*obstacles_ptr)[ii*params->nx + jj] = 1;
                    obs++;
                }
            }

        }
    }
    params->tot_cells = params->nx*params->ny - obs;

    if (accel_area->col_or_row == ACCEL_COLUMN)
    {
         *accelgrid_ptr = (int*) malloc(sizeof(int)*(params->ny));
            if (*accelgrid_ptr == NULL) DIE("Cannot allocate memory for accelgrid_ptr");   
        jj = accel_area->idx;
        for (ii = 0; ii < params->ny; ii++)
        {
            if(!(*obstacles_ptr)[ii*params->nx + jj])
            {
                (*accelgrid_ptr)[ii] = 1;  
            }
            else
            {
                (*accelgrid_ptr)[ii] = 0;
            }

        }
    }
    else
    {
        *accelgrid_ptr = (int*) malloc(sizeof(int)*(params->nx));
            if (*accelgrid_ptr == NULL) DIE("Cannot allocate memory for accelgrid_ptr");   
        ii = accel_area->idx*params->nx;

        for (jj = 0; jj < params->nx; jj++)
        {
            if(!(*obstacles_ptr)[ii + jj])
            {
                (*accelgrid_ptr)[jj] = 1;
            }
            else
            {
                (*accelgrid_ptr)[jj] = 0;
            }
        }
    }
    free(obstacles);
}

void finalise(const param_t* params, speed_t** cells_ptr, speed_t** tmp_cells_ptr,
         int** obstacles_ptr, float** av_vels_ptr, int** accelgrid_ptr, int** rowSetup)
{
  /* 
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  free(*accelgrid_ptr);
  *accelgrid_ptr = NULL;


  free(*rowSetup);
  *rowSetup = NULL;
}

float calc_reynolds(const param_t params, speed_t* cells, int* obstacles, float final_average_velocity)
{
  const float viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);
  
  return final_average_velocity * params.reynolds_dim / viscosity;
}

float total_density(const param_t params, speed_t* cells)
{
  int ii,jj,kk;        /* generic counters */
  float total = 0.0;  /* accumulator */

  for(ii=0;ii<params.ny;ii++) {
    for(jj=0;jj<params.nx;jj++) {
      for(kk=0;kk<NSPEEDS;kk++) {
    total += cells[ii*params.nx + jj].speeds[kk];
      }
    }
  }
  
  return total;
}

void write_values(const param_t params, speed_t* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  int ii,jj,kk;                 /* generic counters */
  const float c_sq = 1.0/3.0;  /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen("final_state.dat","w");
    if (fp == NULL)
    {
        DIE("could not open file output file");
    }

  for(ii=0;ii<params.ny;ii++) {
    for(jj=0;jj<params.nx;jj++) {
      /* an occupied cell */
      if(obstacles[ii*params.nx + jj]) {
    u_x = u_y = u = 0.0;
    pressure = params.density * c_sq;
      }
      /* no obstacle */
      else {
    local_density = 0.0;
    for(kk=0;kk<NSPEEDS;kk++) {
      local_density += cells[ii*params.nx + jj].speeds[kk];
    }
    /* compute x velocity component */
    u_x = (cells[ii*params.nx + jj].speeds[1] + 
           cells[ii*params.nx + jj].speeds[5] +
           cells[ii*params.nx + jj].speeds[8]
           - (cells[ii*params.nx + jj].speeds[3] + 
          cells[ii*params.nx + jj].speeds[6] + 
          cells[ii*params.nx + jj].speeds[7]))
      / local_density;
    /* compute y velocity component */
    u_y = (cells[ii*params.nx + jj].speeds[2] + 
           cells[ii*params.nx + jj].speeds[5] + 
           cells[ii*params.nx + jj].speeds[6]
           - (cells[ii*params.nx + jj].speeds[4] + 
          cells[ii*params.nx + jj].speeds[7] + 
          cells[ii*params.nx + jj].speeds[8]))
      / local_density;
    /* compute norm of velocity */
    u = sqrt((u_x * u_x) + (u_y * u_y));
    /* compute pressure */
    pressure = local_density * c_sq;
      }
      /* write to file */
      fprintf(fp,"%d %d %.12E %.12E %.12E %.12E %d\n",jj,ii,u_x,u_y,u,pressure,obstacles[ii*params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen("av_vels.dat","w");
    if (fp == NULL)
    {
        DIE("could not open file output file");
    }
  for (ii=0;ii<params.max_iters;ii++) {
    fprintf(fp,"%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);
}

void calc_row_setup(int** rowSetup_ptr, int rank, int halorow, int extra, param_t params)
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    *rowSetup_ptr = (int*) malloc(sizeof(int)*3);
        if (*rowSetup_ptr == NULL) DIE("Cannot allocate memory for rowSetup_ptr");

        if(extra == 0) {
            (*rowSetup_ptr)[0] = halorow;
            (*rowSetup_ptr)[1]  = halorow * rank;
            (*rowSetup_ptr)[2]  = halorow * (rank+1);
        } 
        else if(rank < extra) {
            (*rowSetup_ptr)[0]  = halorow;
            (*rowSetup_ptr)[1]  = halorow * rank;
            (*rowSetup_ptr)[2]  = halorow * (rank+1);
        } 
        else {
            (*rowSetup_ptr)[0]  = (params.ny/size + 1) * params.nx;
            (*rowSetup_ptr)[1]  = (halorow + params.nx) * extra + halorow * (rank-extra);
            (*rowSetup_ptr)[2]  = (halorow + params.nx) * extra + halorow * (rank-extra+1);
        } 
    }

void exit_with_error(int line, const char* filename, const char* format, ...)
{
    va_list arglist;

    fprintf(stderr, "Fatal error at line %d in %s: ", line, filename);

    va_start(arglist, format);
    vfprintf(stderr, format, arglist);
    va_end(arglist);

    fprintf(stderr, "\n");

    exit(EXIT_FAILURE);
}