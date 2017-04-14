#ifndef  MATRIX_H
#define  MATRIX_H


#include <iostream>
#include <cassert>
#include <vector>
#include <ostream>
#include <iterator>
#include <algorithm>
#include <cstring>
#include <cmath>


template <class NType>
class Matrix
{
    unsigned int columns, rows;
    NType **data;

    void allocMemory(const unsigned int &rows, const unsigned int &columns)
    {
        data = new NType*[rows];
        for(int i = 0;i < rows;i++)
        {
            data[i] = new NType[columns];
        }

    }

    public:
    Matrix(const unsigned int &rows, const unsigned int &columns, const std::vector<NType> &data, bool columnMajor = false)
    {
        assert(rows * columns == data.size());
        this->columns = columns;
        this->rows = rows;
        allocMemory(rows, columns);
        if(!columnMajor)
        {
            for(int i = 0;i < rows;i++)
            {
                for(int j = 0;j < columns;j++)
                {
                    this->data[i][j] = data[i * columns + j];
                }
            }

        }
        else
        {
            for(int i = 0;i < rows;i++)
            {
                for(int j = 0;j < columns;j++)
                {
                    this->data[i][j] = data[j * rows + i];
                }
            }
        }
    }

    Matrix(const Matrix &m)
    {
        columns = m.columns;
        rows = m.rows;
        allocMemory(rows, columns);
        for(int i = 0;i < rows;i++)
        {
            memcpy(data[i], m.data[i], sizeof(NType) * columns);
        }
    }
//
    Matrix(const unsigned int &rows, const unsigned int &columns)
    {
        this->columns = columns;
        this->rows = rows;
        allocMemory(rows, columns);
        for(int i = 0;i < rows;i++)
            memset(data[i], 0, sizeof(NType) * columns);
    }
//
//
    ~Matrix()
    {
        for(int i = 0;i < rows;i++)
        {
            delete [] data[i];
        }
        delete [] data;
    }
//

    friend std::ostream& operator<<(std::ostream& os, const Matrix &m)
    {
        for(int i = 0;i < m.rows;i++)
        {
            for(int j = 0;j < m.columns;j++)
            {
                os << m.data[i][j] << ' ';
            }
            os << std::endl;
        }

        return os;
    }

    friend Matrix<NType> operator *(const Matrix<NType> &a, const Matrix<NType> &b)
    {

        assert(a.columns == b.rows);
        Matrix newMatrix(a.rows, b.columns);

        for(int i = 0;i < a.rows;i++)
        {
            for(int j = 0;j < b.columns;j++)
            {

                for(int k = 0;k < b.rows;k++)
                {
                        newMatrix.data[i][j] += (a.data[i][k] * b.data[k][j]);
                }

            }
        }
        return newMatrix;
    }
//
//
    Matrix<NType> & operator =(const Matrix<NType> &m)
    {
        if(this == &m)
            return *this;

        this->~Matrix();
        this->columns = m.columns;
        this->rows = m.rows;
        allocMemory(rows, columns);

        for(int i = 0;i < rows;i++)
        {
            memcpy(data[i], m.data[i], sizeof(NType) * columns);
        }
        return *this;
    }
//
    friend Matrix<NType> operator +(const Matrix<NType> &a, const Matrix<NType> &b)
    {
        assert(a.rows == b .rows && a.columns == b.columns);
        Matrix NewMatrix(a);

        for(int i = 0;i < NewMatrix.rows;i++)
        {
            for(int j = 0;j < NewMatrix.columns;j++)
            {
                NewMatrix.data[i][j] += b.data[i][j];
            }
        }
        return NewMatrix;
    }
//
    Matrix<NType> & operator *(const NType &n)
    {
        for(int i = 0;i < this->rows;i++)
        {
            for(int j = 0;j < this->columns;j++)
            {
                this->data[i][j] *= n;
            }
        }
        return *this;
    }
//
//
    Matrix<NType> transpose()
    {

        Matrix<NType> transposedMatrix(columns, rows);
        for(int i = 0;i < rows;i++)
        {
            for(int j = 0;j < columns;j++)
            {
                transposedMatrix.data[j][i] = data[i][j];
            }
        }
        return transposedMatrix;
    }
//
    int getRows()
    {
        return rows;
    }
    int getColumns() const
    {
        return columns;
    }
//
    std::vector<NType> getData() const
    {
		std::vector<NType> vecdata;
		vecdata.reserve(rows * columns);
		for(int i = 0;i < rows;i++)
            for(int j = 0;j < columns;j++)
                vecdata.push_back(data[i][j]);
        return vecdata;
    }
//
    void applyFunction(NType (*func) (NType n))
    {
        for(int i = 0;i < rows;i++)
        {
            for(int j = 0;j < columns;j++)
            {
                data[i][j] = func(data[i][j]);
            }
        }
    }


    Matrix<NType> dataProduct(const Matrix<NType> &m)
    {
        assert(this->rows * this->columns == m.rows * m.columns);
        for(int i = 0;i < m.rows;i++)
        {
            for(int j = 0;j < m.columns;j++)
            {
                this->data[i][j] *= m.data[i][j];
            }
        }
        return *this;
    }
//
    std::vector<NType> getRowData (const unsigned int &row) const
    {
        assert(row < rows);
        std::vector<NType> rowVec;
        for(int i = 0;i < columns;i++)
        {
            rowVec.push_back(data[row][i]);
        }
        return rowVec;
    }
//
	void operator +(const std::vector<NType> &n)
	{
		assert(rows * columns  == n.size());
		for(int i = 0;i < rows;i++)
		{
			for(int j = 0;j < columns;j++)
			{
				data[i][j] += n[i * columns + j];

			}
		}
	}
//
	void reluActivation(const Matrix<NType> &Outputs) const
	{
		assert(rows == Outputs.rows && columns == Outputs.columns);
		for(int i = 0;i < rows;i++)
		{
			for(int j = 0;j < columns;j++)
			{
				if(Outputs.data[i][j] == 0)
				{
					data[i][j] = 0;
				}
			}
		}
	}
//
	void sigmoidActivation(const Matrix<NType> &Outputs) const
	{
		assert(rows == Outputs.rows && columns == Outputs.columns);
		for(int i = 0;i < rows;i++)
		{
			for(int j = 0;j < columns;j++)
			{
			    //cout << data[i * columns + j] << endl << (Outputs.data[i * columns + j]*(1 - Outputs.data[i * columns + j])) << endl;
				data[i][j] *= (Outputs.data[i][j]*(1 - Outputs.data[i][j]));
			}
		}
	}
//
	void multiplyRow(NType constant, const unsigned int &row)
	{
	    assert(row < rows);
	    for(int i = 0;i < columns;i++)
        {
            data[row][i] *= constant;
        }
	}
//
	void dimensions()
	{
	    std::cout << "ROWS: " << rows << std::endl << "Columns: " << columns << std::endl;
	}
//
//
	friend Matrix<NType> hardToNameThis(const Matrix<NType> &a, const Matrix<NType> &b)
	{
	    assert(a.rows == 1 && b.rows == 1);
	    std::vector<NType> data;
	    std::vector<NType> temp = b.getRowData(0);
	    for(int i = 0;i < a.columns;i++)
        {
            data.insert(data.end(), temp.begin(), temp.end());
        }
        Matrix<NType> m(a.columns, b.columns, data);
        // cout << b.columns << endl << a.columns;
        for(int i = 0;i < m.rows;i++)
        {

                m.multiplyRow(a.data[0][i], i);

        }
        return m;
	}
	//! Applies the softmax activation function to a given one row Matrix
	/*!
		\param expSum - Precomputed exponential sum of the output layers net input.
	*/
	void softmaxOutputActivation(const NType &expSum) 
	{
		for(int i = 0;i < columns;i++)
		{
			data[0][i] = exp(data[0][i])/expSum;
		}
	}

};

#endif
