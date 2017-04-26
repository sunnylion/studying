#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <mpi.h>

#define ind(i, j) ((i + 1) * l->ny + ((j + l->ny) % l->ny))

typedef struct {
	int nx, ny;
	int *u0;
	int *u1;
	int steps;
	int save_steps;
} life_t;

int size, rank;
MPI_Status status;

void life_init(const char *path, life_t *l);
void life_free(life_t *l);
void life_step(life_t *l);
void life_save_vtk(const char *path, life_t *l);
void exchange(life_t *l);

int main(int argc, char **argv)
{
	if (argc != 2) {
		printf("Usage: %s input file.\n", argv[0]);
		return 0;
	}
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	life_t l;
	life_init(argv[1], &l);
	
	int i;
	char buf[100];
	for (i = 0; i < l.steps; i++) {
		if (i % l.save_steps == 0) {
			sprintf(buf, "life_%06d.vtk", i);
			printf("Saving step %d to '%s'.\n", i, buf);
			if(rank==0)
				life_save_vtk(buf, &l);
		}
		life_step(&l);
		exchange(&l);
	}
	
	life_free(&l);
	MPI_Finalize();
	return 0;
}

void exchange(life_t *l)
{
	int right = (rank + 1) % size;
	int left = (size + rank - 1) % size;
	MPI_Sendrecv( l->u0 + ind(l->nx - 1,0), l->ny, MPI_INT, right, rank, l->u0, l->ny, MPI_INT, left, left, MPI_COMM_WORLD, &status );
	MPI_Sendrecv( l->u0 + ind(0,0), l->ny, MPI_INT, left, rank, l->u0 + ind(l->nx, 0), l->ny, MPI_INT, right, right, MPI_COMM_WORLD, &status );
}

/**
 * Загрузить входную конфигурацию.
 * Формат файла, число шагов, как часто сохранять, размер поля, затем идут координаты заполненых клеток:
 * steps
 * save_steps
 * nx ny
 * i1 j2
 * i2 j2
 */
void life_init(const char *path, life_t *l)
{

	FILE *fd = fopen(path, "r");
	assert(fd);
	assert(fscanf(fd, "%d\n", &l->steps));
	assert(fscanf(fd, "%d\n", &l->save_steps));
	printf("Steps %d, save every %d step.\n", l->steps, l->save_steps);
	assert(fscanf(fd, "%d %d\n", &l->nx, &l->ny));
	printf("Field size: %dx%d\n", l->nx, l->ny);
	
	int fullWidth = l->nx;
	int averageWidth = l->nx/size;
	l->nx = l->nx/size + ((rank + 1) / size) * (l->nx % size);

	l->u0 = (int*)calloc((l->nx + 2) * l->ny, sizeof(int));
	l->u1 = (int*)calloc((l->nx + 2) * l->ny, sizeof(int));
	
	int i, j, r, cnt;
	cnt = 0;
	while ((r = fscanf(fd, "%d %d\n", &i, &j)) != EOF) {
		if(i == fullWidth - 1 && rank == 0)
			i = -1;
		else if(i == 0 && rank == size - 1)
			i = l->nx;
		else
			i = i - averageWidth * rank;
			
		if(i >= -1 && i <= l->nx ){
			l->u0[ind(i, j)] = 1;
			cnt++;
		}
	}
	printf("Loaded %d life cells.\n", cnt);
	fclose(fd);
}

void life_free(life_t *l)
{
	free(l->u0);
	free(l->u1);
	l->nx = l->ny = 0;
}

void life_save_vtk(const char *path, life_t *l)
{
	FILE *f;
	int i1, i2, j;
	f = fopen(path, "w");
	assert(f);
	fprintf(f, "# vtk DataFile Version 3.0\n");
	fprintf(f, "Created by write_to_vtk2d\n");
	fprintf(f, "ASCII\n");
	fprintf(f, "DATASET STRUCTURED_POINTS\n");
	fprintf(f, "DIMENSIONS %d %d 1\n", l->nx+1, l->ny+1);
	fprintf(f, "SPACING %d %d 0.0\n", 1, 1);
	fprintf(f, "ORIGIN %d %d 0.0\n", 0, 0);
	fprintf(f, "CELL_DATA %d\n", l->nx * l->ny);
	
	fprintf(f, "SCALARS life int 1\n");
	fprintf(f, "LOOKUP_TABLE life_table\n");
	for (i2 = 0; i2 < l->ny; i2++) {
		for (i1 = 0; i1 < l->nx; i1++) {
			fprintf(f, "%d\n", l->u0[ind(i1, i2)]);
		}
	}
	fclose(f);
}

void life_step(life_t *l)
{
	int i, j;
	for (j = 0; j < l->ny; j++) {
		for (i = 0; i < l->nx; i++) {
			int n = 0;
			n += l->u0[ind(i+1, j)];
			n += l->u0[ind(i+1, j+1)];
			n += l->u0[ind(i,   j+1)];
			n += l->u0[ind(i-1, j)];
			n += l->u0[ind(i-1, j-1)];
			n += l->u0[ind(i,   j-1)];
			n += l->u0[ind(i-1, j+1)];
			n += l->u0[ind(i+1, j-1)];
			l->u1[ind(i,j)] = 0;
			if (n == 3 && l->u0[ind(i,j)] == 0) {
				l->u1[ind(i,j)] = 1;
			}
			if ((n == 3 || n == 2) && l->u0[ind(i,j)] == 1) {
				l->u1[ind(i,j)] = 1;
			}
		}
	}
	int *tmp;
	tmp = l->u0;
	l->u0 = l->u1;
	l->u1 = tmp;
}


