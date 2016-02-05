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


void accelerate_flow(const param_t params, const accel_area_t accel_area, speed_t* cells, int start_row, int end_row, int* accelgrid)
{

    int ii,jj;  
    int rank, size;  
    // int          blocklen[1] = {9}; 
    // MPI_Datatype type[1] = {MPI_FLOAT};
    // MPI_Datatype Speedtype; 
    // MPI_Aint     disp[1]; 
    // disp[0] = 0;
 
    // MPI_Type_create_struct( 1, blocklen, disp, type, &Speedtype); 
    // MPI_Type_commit( &Speedtype); 

    int up, down;
    float* recv_up      = malloc(sizeof(float) * params.nx * 3);
    float* recv_down    = malloc(sizeof(float) * params.nx * 3);
    float* send_up        = malloc(sizeof(float) * params.nx * 3);
    float* send_down      = malloc(sizeof(float) * params.nx * 3);
    /* compute weighting factors */ 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (accel_area.col_or_row == ACCEL_COLUMN)
        {
            jj = accel_area.idx;
            for (ii = 0; ii < params.ny; ii++)
            {
                    cells[ii*params.nx + jj].speeds[2] += accelgrid[ii]*0.0000555;
                    cells[ii*params.nx + jj].speeds[5] += accelgrid[ii]*0.0000138;
                    cells[ii*params.nx + jj].speeds[6] += accelgrid[ii]*0.0000138;

                    cells[ii*params.nx + jj].speeds[4] -= accelgrid[ii]*0.0000555;
                    cells[ii*params.nx + jj].speeds[7] -= accelgrid[ii]*0.0000138;
                    cells[ii*params.nx + jj].speeds[8] -= accelgrid[ii]*0.0000138; 
            }
        }
        else
        {
            ii = accel_area.idx*params.nx;
            for (jj = 0; jj < params.nx; jj++)
            {
                    cells[ii + jj].speeds[1] += accelgrid[jj]*0.0000555;
                    cells[ii + jj].speeds[5] += accelgrid[jj]*0.0000138;
                    cells[ii + jj].speeds[8] += accelgrid[jj]*0.0000138;
                    
                    cells[ii + jj].speeds[3] -= accelgrid[jj]*0.0000555;
                    cells[ii + jj].speeds[6] -= accelgrid[jj]*0.0000138;
                    cells[ii + jj].speeds[7] -= accelgrid[jj]*0.0000138;
            }
        }
        
    //}
    // else
    // {
    //     MPI_Bcast(cells, params.nx*params.ny, Speedtype, rank, MPI_COMM_WORLD);
    // // }
    // MPI_Bcast(&cells, params.nx*params.ny, Speedtype, rank, MPI_COMM_WORLD);
    // MPI_Barrier(MPI_COMM_WORLD);


    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(size > 1)
    {
        int rank;
        MPI_Status status;  

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank == 0) {
        up = 1;
        down = size-1;
        } else if(rank == size-1) {
            up = 0;
            down = size-2;
        } else {
            up = rank+1;
            down = rank-1;
        }
        for(ii=0; ii<params.nx; ii++) {
            send_down[ii*3]    = cells[start_row+ii].speeds[4];
            send_down[ii*3+1]  = cells[start_row+ii].speeds[7];
            send_down[ii*3+2]  = cells[start_row+ii].speeds[8];

            send_up[ii*3]      = cells[end_row-params.nx+ii].speeds[2];
            send_up[ii*3+1]    = cells[end_row-params.nx+ii].speeds[5];
            send_up[ii*3+2]    = cells[end_row-params.nx+ii].speeds[6];
        }

        /* send down, receive from up */
        MPI_Sendrecv(send_down, params.nx * 3, MPI_FLOAT, down, 1,
                    recv_up, params.nx * 3, MPI_FLOAT, up, 1,
                    MPI_COMM_WORLD, &status);

        // send up, receive from down 
        MPI_Sendrecv(send_up, params.nx * 3, MPI_FLOAT, up, 2,
                    recv_down, params.nx * 3, MPI_FLOAT, down, 2,
                    MPI_COMM_WORLD, &status);

        if(rank == 0) {
            for(ii=0; ii<params.nx; ii++) {
                cells[end_row+ii].speeds[4] = recv_up[ii*3];
                cells[end_row+ii].speeds[7] = recv_up[ii*3+1];
                cells[end_row+ii].speeds[8] = recv_up[ii*3+2];

                cells[params.nx*(params.ny-1)+ii].speeds[2] = recv_down[ii*3];
                cells[params.nx*(params.ny-1)+ii].speeds[5] = recv_down[ii*3+1];
                cells[params.nx*(params.ny-1)+ii].speeds[6] = recv_down[ii*3+2];
            }
        } else if(rank == size-1) {
            for(ii=0; ii<params.nx; ii++) {
                cells[ii].speeds[4] = recv_up[ii*3];
                cells[ii].speeds[7] = recv_up[ii*3+1];
                cells[ii].speeds[8] = recv_up[ii*3+2];

                cells[start_row-params.nx+ii].speeds[2] = recv_down[ii*3];
                cells[start_row-params.nx+ii].speeds[5] = recv_down[ii*3+1];
                cells[start_row-params.nx+ii].speeds[6] = recv_down[ii*3+2];
            }
        } else {
            for(ii=0; ii<params.nx; ii++) {
                cells[end_row+ii].speeds[4] = recv_up[ii*3];
                cells[end_row+ii].speeds[7] = recv_up[ii*3+1];
                cells[end_row+ii].speeds[8] = recv_up[ii*3+2];

                cells[start_row-params.nx+ii].speeds[2] = recv_down[ii*3];
                cells[start_row-params.nx+ii].speeds[5] = recv_down[ii*3+1];
                cells[start_row-params.nx+ii].speeds[6] = recv_down[ii*3+2];
            }
        }

        free(recv_up);
        free(recv_down);
        free(send_up);
        free(send_down);

        recv_up = NULL;
        recv_down = NULL;
        send_up = NULL;
        send_down = NULL;
    }
    //MPI_Type_free(&Speedtype);
}

void lattice(const param_t params, speed_t* cells, speed_t* tmp_cells, int* obstacles, float* av_vels, int start_row, int end_row, int stepnum)
{
    int ii;                    /* generic counters */
    float tot_u = 0.0, u_sq, u_x,u_y;           /* accumulated magnitudes of velocity for each cell */
    // MPI variables.
    int size;
    int rank;
    float local_density = 0.0;

    // Get number of processes.
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Determine rank of current process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 0) {

        tmp_cells[0].speeds[0] = cells[0].speeds[0];
        tmp_cells[0].speeds[1] = cells[params.nx-1].speeds[1];
        tmp_cells[0].speeds[2] = cells[params.nx*(params.ny-1)].speeds[2];
        tmp_cells[0].speeds[3] = cells[1].speeds[3];
        tmp_cells[0].speeds[4] = cells[params.nx].speeds[4];
        tmp_cells[0].speeds[5] = cells[params.nx*params.ny-1].speeds[5];
        tmp_cells[0].speeds[6] = cells[params.nx*(params.ny-1)+1].speeds[6];
        tmp_cells[0].speeds[7] = cells[params.nx+1].speeds[7];
        tmp_cells[0].speeds[8] = cells[2*params.nx-1].speeds[8];

        tmp_cells[params.nx-1].speeds[0] = cells[params.nx-1].speeds[0];
        tmp_cells[params.nx-1].speeds[1] = cells[params.nx-2].speeds[1];
        tmp_cells[params.nx-1].speeds[2] = cells[(params.nx-1)*(params.ny-1)].speeds[2];
        tmp_cells[params.nx-1].speeds[3] = cells[0].speeds[3];
        tmp_cells[params.nx-1].speeds[4] = cells[2*params.nx-1].speeds[4];
        tmp_cells[params.nx-1].speeds[5] = cells[params.nx*params.ny-2].speeds[5];
        tmp_cells[params.nx-1].speeds[6] = cells[params.nx*(params.ny-1)].speeds[6];
        tmp_cells[params.nx-1].speeds[7] = cells[params.nx].speeds[7];
        tmp_cells[params.nx-1].speeds[8] = cells[2*params.nx-2].speeds[8];

        for(ii=1;ii<params.nx-1;ii++) {
            tmp_cells[ii].speeds[0] = cells[ii].speeds[0];
            tmp_cells[ii].speeds[1] = cells[ii-1].speeds[1];
            tmp_cells[ii].speeds[2] = cells[params.nx*(params.ny-1)+ii].speeds[2];
            tmp_cells[ii].speeds[3] = cells[ii+1].speeds[3];
            tmp_cells[ii].speeds[4] = cells[ii+params.nx].speeds[4];
            tmp_cells[ii].speeds[5] = cells[params.nx*(params.ny-1)+ii-1].speeds[5];
            tmp_cells[ii].speeds[6] = cells[params.nx*(params.ny-1)+ii+1].speeds[6];
            tmp_cells[ii].speeds[7] = cells[params.nx+ii+1].speeds[7];
            tmp_cells[ii].speeds[8] = cells[params.nx+ii-1].speeds[8];
        }
    }

    if(rank == size-1) {

        tmp_cells[params.nx*(params.ny-1)].speeds[0] = cells[params.nx*(params.ny-1)].speeds[0];
        tmp_cells[params.nx*(params.ny-1)].speeds[1] = cells[params.nx*params.ny-1].speeds[1];
        tmp_cells[params.nx*(params.ny-1)].speeds[2] = cells[params.nx*(params.ny-2)].speeds[2];
        tmp_cells[params.nx*(params.ny-1)].speeds[3] = cells[params.nx*(params.ny-1)+1].speeds[3];
        tmp_cells[params.nx*(params.ny-1)].speeds[4] = cells[0].speeds[4];
        tmp_cells[params.nx*(params.ny-1)].speeds[5] = cells[params.nx*(params.ny-1)-1].speeds[5];
        tmp_cells[params.nx*(params.ny-1)].speeds[6] = cells[params.nx*(params.ny-2)+1].speeds[6];
        tmp_cells[params.nx*(params.ny-1)].speeds[7] = cells[1].speeds[7];


        tmp_cells[params.nx*params.ny-1].speeds[0] = cells[params.nx*params.ny-1].speeds[0];
        tmp_cells[params.nx*params.ny-1].speeds[1] = cells[params.nx*params.ny-2].speeds[1];
        tmp_cells[params.nx*params.ny-1].speeds[2] = cells[params.nx*(params.ny-1)-1].speeds[2];
        tmp_cells[params.nx*params.ny-1].speeds[3] = cells[params.nx*(params.ny-1)].speeds[3];
        tmp_cells[params.nx*params.ny-1].speeds[4] = cells[params.nx-1].speeds[4];
        tmp_cells[params.nx*params.ny-1].speeds[5] = cells[params.nx*(params.ny-1)-2].speeds[5];
        tmp_cells[params.nx*params.ny-1].speeds[6] = cells[params.nx*(params.ny-2)].speeds[6];
        tmp_cells[params.nx*params.ny-1].speeds[7] = cells[0].speeds[7];
        tmp_cells[params.nx*params.ny-1].speeds[8] = cells[params.nx-2].speeds[8];


        for(ii=params.nx*(params.ny-1)+1;ii<params.ny*params.nx-1;ii++) {
            tmp_cells[ii].speeds[0] = cells[ii].speeds[0];
            tmp_cells[ii].speeds[1] = cells[ii-1].speeds[1];
            tmp_cells[ii].speeds[2] = cells[ii-params.nx].speeds[2];
            tmp_cells[ii].speeds[3] = cells[ii+1].speeds[3];
            tmp_cells[ii].speeds[4] = cells[ii-params.nx*(params.ny-1)].speeds[4];
            tmp_cells[ii].speeds[5] = cells[ii-params.nx-1].speeds[5];
            tmp_cells[ii].speeds[6] = cells[ii-params.nx+1].speeds[6];
            tmp_cells[ii].speeds[7] = cells[ii-params.nx*(params.ny-1)+1].speeds[7];
            tmp_cells[ii].speeds[8] = cells[ii-params.nx*(params.ny-1)-1].speeds[8];
        }
    }

    int local_start = (rank == 0) ? params.nx : start_row;
    int local_end = (rank == size-1)? end_row-params.nx: end_row;
    for(ii=local_start; ii<local_end; ii++) {

        if(ii%params.nx==0) {
            tmp_cells[ii].speeds[0] = cells[ii].speeds[0];
            tmp_cells[ii].speeds[1] = cells[ii+params.nx-1].speeds[1];
            tmp_cells[ii].speeds[2] = cells[ii-params.nx].speeds[2];
            tmp_cells[ii].speeds[3] = cells[ii+1].speeds[3];
            tmp_cells[ii].speeds[4] = cells[ii+params.nx].speeds[4];
            tmp_cells[ii].speeds[5] = cells[ii-1].speeds[5];
            tmp_cells[ii].speeds[6] = cells[ii-params.nx+1].speeds[6];
            tmp_cells[ii].speeds[7] = cells[ii+params.nx+1].speeds[7];
            tmp_cells[ii].speeds[8] = cells[ii+2*params.nx-1].speeds[8];
            continue;
        }

        if((ii+1)%params.nx==0) {
            tmp_cells[ii].speeds[0] = cells[ii].speeds[0];
            tmp_cells[ii].speeds[1] = cells[ii-1].speeds[1];
            tmp_cells[ii].speeds[2] = cells[ii-params.nx].speeds[2];
            tmp_cells[ii].speeds[3] = cells[ii-params.nx+1].speeds[3];
            tmp_cells[ii].speeds[4] = cells[ii+params.nx].speeds[4];
            tmp_cells[ii].speeds[5] = cells[ii-params.nx-1].speeds[5];
            tmp_cells[ii].speeds[6] = cells[ii-2*params.nx+1].speeds[6];
            tmp_cells[ii].speeds[7] = cells[ii+1].speeds[7];
            tmp_cells[ii].speeds[8] = cells[ii+params.nx-1].speeds[8];
            continue;
        }

        tmp_cells[ii].speeds[0] = cells[ii].speeds[0];
        tmp_cells[ii].speeds[1] = cells[ii-1].speeds[1];
        tmp_cells[ii].speeds[2] = cells[ii-params.nx].speeds[2];
        tmp_cells[ii].speeds[3] = cells[ii+1].speeds[3];
        tmp_cells[ii].speeds[4] = cells[ii+params.nx].speeds[4];
        tmp_cells[ii].speeds[5] = cells[ii-params.nx-1].speeds[5];
        tmp_cells[ii].speeds[6] = cells[ii-params.nx+1].speeds[6];
        tmp_cells[ii].speeds[7] = cells[ii+params.nx+1].speeds[7];
        tmp_cells[ii].speeds[8] = cells[ii+params.nx-1].speeds[8];
    }

    for(ii=start_row; ii<end_row; ii++) {

        if(!obstacles[ii]) {
            
            local_density = tmp_cells[ii].speeds[0]+tmp_cells[ii].speeds[1]+tmp_cells[ii].speeds[2]+tmp_cells[ii].speeds[3]
                           +tmp_cells[ii].speeds[4]+tmp_cells[ii].speeds[5]+tmp_cells[ii].speeds[6]+tmp_cells[ii].speeds[7]+tmp_cells[ii].speeds[8];
                 /* compute x velocity component */
            u_x = (tmp_cells[ii].speeds[1] +tmp_cells[ii].speeds[5] +tmp_cells[ii].speeds[8]- (tmp_cells[ii].speeds[3] 
                   +tmp_cells[ii].speeds[6] +tmp_cells[ii].speeds[7]))/ local_density;
                 /* compute y velocity component */
            u_y = (tmp_cells[ii].speeds[2] +tmp_cells[ii].speeds[5] +tmp_cells[ii].speeds[6]- (tmp_cells[ii].speeds[4] 
                   +tmp_cells[ii].speeds[7] +tmp_cells[ii].speeds[8]))/ local_density;
              
            /* accumulate the norm of x- and y- velocity components */
            tot_u += sqrt((u_x * u_x) + (u_y * u_y));

            /* velocity squared */ 
            u_sq = u_x * u_x + u_y * u_y;

            /* relaxation step */
            cells[ii].speeds[0] = (tmp_cells[ii].speeds[0]+ params.omega * (0.44444444 * local_density * (1.0 - u_sq * 1.5) - tmp_cells[ii].speeds[0]));
            cells[ii].speeds[1] = (tmp_cells[ii].speeds[1]+ params.omega * (0.11111111 * local_density * (1.0 + u_x * 3.0+ (u_x * u_x) * 4.5- u_sq * 1.5)- tmp_cells[ii].speeds[1]));
            cells[ii].speeds[2] = (tmp_cells[ii].speeds[2]+ params.omega * (0.11111111 * local_density * (1.0 + u_y * 3.0+ (u_y * u_y) * 4.5- u_sq * 1.5) - tmp_cells[ii].speeds[2]));
            cells[ii].speeds[3] = (tmp_cells[ii].speeds[3]+ params.omega * (0.11111111 * local_density * (1.0 - u_x * 3.0+ (u_x * u_x) * 4.5- u_sq * 1.5) - tmp_cells[ii].speeds[3]));
            cells[ii].speeds[4] = (tmp_cells[ii].speeds[4]+ params.omega * (0.11111111 * local_density * (1.0 - u_y * 3.0+ (u_y * u_y) * 4.5- u_sq * 1.5) - tmp_cells[ii].speeds[4]));
            cells[ii].speeds[5] = (tmp_cells[ii].speeds[5]+ params.omega * (0.027777777 * local_density * (1.0 + (u_x + u_y) * 3.0+ ((u_x + u_y) * (u_x + u_y)) * 4.5- u_sq * 1.5) - tmp_cells[ii].speeds[5]));
            cells[ii].speeds[6] = (tmp_cells[ii].speeds[6]+ params.omega * (0.027777777 * local_density * (1.0 + (- u_x + u_y) * 3.0+ ((- u_x + u_y)* (- u_x + u_y)) * 4.5- u_sq * 1.5) - tmp_cells[ii].speeds[6]));
            cells[ii].speeds[7] = (tmp_cells[ii].speeds[7]+ params.omega * (0.027777777 * local_density * (1.0 + (- u_x - u_y) * 3.0+ ((- u_x - u_y) * (- u_x - u_y)) * 4.5- u_sq * 1.5) - tmp_cells[ii].speeds[7]));
            cells[ii].speeds[8] = (tmp_cells[ii].speeds[8]+ params.omega * (0.027777777 * local_density * (1.0 + (u_x - u_y) * 3.0+ ((u_x - u_y) * (u_x - u_y)) * 4.5- u_sq * 1.5) - tmp_cells[ii].speeds[8]));
        } else {

            cells[ii].speeds[1] = tmp_cells[ii].speeds[3];
            cells[ii].speeds[2] = tmp_cells[ii].speeds[4];
            cells[ii].speeds[3] = tmp_cells[ii].speeds[1];
            cells[ii].speeds[4] = tmp_cells[ii].speeds[2];
            cells[ii].speeds[5] = tmp_cells[ii].speeds[7];
            cells[ii].speeds[6] = tmp_cells[ii].speeds[8];
            cells[ii].speeds[7] = tmp_cells[ii].speeds[5];
            cells[ii].speeds[8] = tmp_cells[ii].speeds[6];
        }
    }

    float total_u = 0.0;
    MPI_Reduce(&tot_u, &total_u, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        av_vels[stepnum] = total_u / (float)params.tot_cells;
    }
}