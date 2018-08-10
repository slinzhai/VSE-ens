package util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class Lists {
	/**
	 * sort an {@code Map<K, V extends Comparable<? extends V>} map object
	 * 
	 * <p>
	 * <strong>Remark: </strong> note that this method may be memory-consuming as it needs to make an ArrayList copy of
	 * input Map data. Instead, we suggest to store original data in List<Map.Entry<K,V>> and use sortList() method to
	 * avoid object copying.
	 * </p>
	 * 
	 * @param data
	 *            map data
	 * @param inverse
	 *            descending if true; otherwise ascending
	 * @return a sorted list
	 * 
	 */
	public static <K, V extends Comparable<? super V>> List<Map.Entry<K, V>> sortMap(Map<K, V> data, final boolean inverse) {
		// According to tests, LinkedList is slower than ArrayList
		List<Map.Entry<K, V>> pairs = new ArrayList<>(data.entrySet());
		sortList(pairs, inverse);
		return pairs;
	}

	/**
	 * sort a map object: {@code Map<K, V extends Comparable<? extends V>}
	 * 
	 * @param data
	 *            map data
	 * @return an ascending sorted list
	 */
	public static <K, V extends Comparable<? super V>> List<Map.Entry<K, V>> sortMap(Map<K, V> data) {
		return sortMap(data, false);
	}
	/**
	 * sort a list of objects: {@code List<Map.Entry<K, V extends Comparable<? extends V>>}
	 * 
	 * @param data
	 *            map data
	 * @param inverse
	 *            descending if true; otherwise ascending
	 * @return a sorted list
	 */
	public static <K, V extends Comparable<? super V>> void sortList(List<Map.Entry<K, V>> data, final boolean inverse) {
		Collections.sort(data, new Comparator<Map.Entry<K, V>>() {
			@Override
			public int compare(Entry<K, V> a, Entry<K, V> b) {
				int res = (a.getValue()).compareTo(b.getValue());
				return inverse ? -res : res;
			}
		});
	}

	/**
	 * sort a map object: {@code List<Map.Entry<K, V extends Comparable<? extends V>>}
	 * 
	 * @param data
	 *            map data
	 * @return an ascending sorted list
	 */
	public static <K, V extends Comparable<? super V>> void sortList(List<Map.Entry<K, V>> data) {
		sortList(data, false);

	}
}
